// Function: sub_1B7DC70
// Address: 0x1b7dc70
//
__int64 __fastcall sub_1B7DC70(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rax
  _QWORD *v5; // r12
  double v6; // xmm0_8
  __int64 v7; // rdx
  __int64 v8; // rsi
  unsigned __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 *v11; // rax
  __int64 v12; // r12
  __int64 *v13; // rax
  __int64 *v14; // rax
  int v15; // [rsp-34h] [rbp-34h] BYREF
  __int64 v16; // [rsp-30h] [rbp-30h] BYREF

  v2 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v2 + 16) )
    BUG();
  result = *(unsigned int *)(v2 + 36);
  if ( (_DWORD)result != 4085 && (_DWORD)result != 4503 )
  {
    if ( (_DWORD)result == 4057 )
    {
      v11 = (__int64 *)sub_16498A0(a1);
      v15 = 1;
      v12 = sub_155D330(v11, a2);
    }
    else
    {
      if ( (_DWORD)result != 4492 )
        return result;
      v14 = (__int64 *)sub_16498A0(a1);
      v15 = 2;
      v12 = sub_155D330(v14, a2);
    }
    v16 = *(_QWORD *)(a1 + 56);
    v13 = (__int64 *)sub_16498A0(a1);
    result = sub_1563E10(&v16, v13, &v15, 1, v12);
    *(_QWORD *)(a1 + 56) = result;
    return result;
  }
  v4 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v5 = *(_QWORD **)(v4 + 24);
  if ( *(_DWORD *)(v4 + 32) > 0x40u )
    v5 = (_QWORD *)*v5;
  v6 = log2((double)(int)a2);
  v7 = sub_15A0680(
         **(_QWORD **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)),
         (unsigned int)v5 & 0xFFFC1FFF | (((int)(v6 + 1.0) & 0x1F) << 13),
         0);
  result = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( *(_QWORD *)result )
  {
    v8 = *(_QWORD *)(result + 8);
    v9 = *(_QWORD *)(result + 16) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v9 = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(v8 + 16) & 3LL | v9;
  }
  *(_QWORD *)result = v7;
  if ( v7 )
  {
    v10 = *(_QWORD *)(v7 + 8);
    *(_QWORD *)(result + 8) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = (result + 8) | *(_QWORD *)(v10 + 16) & 3LL;
    *(_QWORD *)(result + 16) = (v7 + 8) | *(_QWORD *)(result + 16) & 3LL;
    *(_QWORD *)(v7 + 8) = result;
  }
  return result;
}

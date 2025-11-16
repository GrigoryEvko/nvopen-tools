// Function: sub_F6EC60
// Address: 0xf6ec60
//
__int64 __fastcall sub_F6EC60(__int64 a1, _DWORD *a2)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v7; // rsi
  _QWORD *v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // rdx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rax
  int v13; // edx
  bool v14; // cc
  int v15; // eax
  unsigned __int64 v16; // [rsp+0h] [rbp-30h] BYREF
  __int64 v17; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_F6C0B0(a1);
  if ( !v4 )
    goto LABEL_3;
  v5 = v4;
  if ( !(unsigned __int8)sub_BC8C50(v4, &v16, &v17) )
    goto LABEL_3;
  v7 = *(_QWORD *)(v5 - 64);
  if ( !*(_BYTE *)(a1 + 84) )
  {
    if ( !sub_C8CA60(a1 + 56, v7) )
    {
      v11 = v17;
      goto LABEL_10;
    }
    v9 = v17;
LABEL_9:
    v11 = v16;
    v16 = v9;
    v17 = v11;
    goto LABEL_10;
  }
  v8 = *(_QWORD **)(a1 + 64);
  v9 = v17;
  v10 = &v8[*(unsigned int *)(a1 + 76)];
  v11 = v17;
  if ( v8 != v10 )
  {
    while ( v7 != *v8 )
    {
      if ( v10 == ++v8 )
        goto LABEL_10;
    }
    goto LABEL_9;
  }
LABEL_10:
  if ( !v11 )
  {
LABEL_3:
    BYTE4(v17) = 0;
    return v17;
  }
  v12 = (v16 % v11 > (v11 - 1) >> 1) + v16 / v11;
  v13 = v12 + 1;
  v14 = v12 <= 0xFFFFFFFE;
  v15 = -1;
  if ( v14 )
    v15 = v13;
  if ( a2 )
    *a2 = v11;
  LODWORD(v17) = v15;
  BYTE4(v17) = 1;
  return v17;
}

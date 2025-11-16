// Function: sub_18F3A80
// Address: 0x18f3a80
//
__int64 __fastcall sub_18F3A80(__int64 a1, _QWORD **a2)
{
  _QWORD *v3; // rbx
  __int64 v4; // r15
  char v5; // al
  __int64 v6; // rdx
  bool v7; // zf
  __int64 v8; // rax
  __int64 *v9; // rax
  char v10; // r8
  __int64 result; // rax
  __int64 v12; // rdx
  int v13; // ecx
  __int64 v14; // r8
  int v15; // edi
  unsigned int v16; // eax
  _QWORD *v17; // rsi
  _QWORD *v18; // r9
  unsigned int v19; // eax
  int v20; // esi
  int v21; // r10d
  __int64 v22; // [rsp+8h] [rbp-78h]
  unsigned __int8 v23; // [rsp+1Fh] [rbp-61h]
  _QWORD v24[12]; // [rsp+20h] [rbp-60h] BYREF

  v3 = *a2;
  v4 = *(_QWORD *)a1;
  v22 = **(_QWORD **)(a1 + 8);
  v23 = sub_15E4690(**(_QWORD **)(a1 + 16), 0);
  v5 = sub_140E950(v3, v24, v4, v22, v23 << 16);
  v24[2] = 0;
  v6 = *(_QWORD *)(a1 + 32);
  v7 = v5 == 0;
  v8 = -1;
  if ( !v7 )
    v8 = v24[0];
  v24[3] = 0;
  v24[0] = v3;
  v24[1] = v8;
  v9 = *(__int64 **)(a1 + 24);
  v24[4] = 0;
  v10 = sub_134CB50(*v9, (__int64)v24, v6);
  result = 0;
  if ( v10 )
  {
    v12 = *(_QWORD *)(a1 + 40);
    if ( (*(_BYTE *)(v12 + 8) & 1) != 0 )
    {
      v14 = v12 + 16;
      v15 = 15;
    }
    else
    {
      v13 = *(_DWORD *)(v12 + 24);
      v14 = *(_QWORD *)(v12 + 16);
      result = 1;
      if ( !v13 )
        return result;
      v15 = v13 - 1;
    }
    v16 = v15 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v17 = (_QWORD *)(v14 + 8LL * v16);
    v18 = (_QWORD *)*v17;
    if ( *a2 == (_QWORD *)*v17 )
    {
LABEL_9:
      *v17 = -16;
      v19 = *(_DWORD *)(v12 + 8);
      ++*(_DWORD *)(v12 + 12);
      *(_DWORD *)(v12 + 8) = (2 * (v19 >> 1) - 2) | v19 & 1;
      return 1;
    }
    else
    {
      v20 = 1;
      while ( v18 != (_QWORD *)-8LL )
      {
        v21 = v20 + 1;
        v16 = v15 & (v20 + v16);
        v17 = (_QWORD *)(v14 + 8LL * v16);
        v18 = (_QWORD *)*v17;
        if ( *a2 == (_QWORD *)*v17 )
          goto LABEL_9;
        v20 = v21;
      }
      return 1;
    }
  }
  return result;
}

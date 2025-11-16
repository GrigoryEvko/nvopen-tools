// Function: sub_AFCD70
// Address: 0xafcd70
//
__int64 __fastcall sub_AFCD70(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r13d
  __int64 v6; // r15
  __int64 v7; // r14
  unsigned __int8 v9; // al
  __int64 *v10; // rdx
  int v11; // eax
  int v12; // r13d
  __int64 v13; // rsi
  int v14; // r8d
  _QWORD *v15; // rdi
  unsigned int v16; // eax
  _QWORD *v17; // rcx
  __int64 v18; // rdx
  int v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h] BYREF
  bool v23; // [rsp+28h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v21 = *(_DWORD *)(*a2 + 24);
    if ( v21 > 0x40 )
      sub_C43780(&v20, v6 + 16);
    else
      v20 = *(_QWORD *)(v6 + 16);
    v9 = *(_BYTE *)(v6 - 16);
    if ( (v9 & 2) != 0 )
      v10 = *(__int64 **)(v6 - 32);
    else
      v10 = (__int64 *)(v6 - 16 - 8LL * ((v9 >> 2) & 0xF));
    v22 = *v10;
    v23 = *(_DWORD *)(v6 + 4) != 0;
    v11 = sub_AFB7E0((__int64)&v20, &v22);
    if ( v21 > 0x40 && v20 )
    {
      v19 = v11;
      j_j___libc_free_0_0(v20);
      v11 = v19;
    }
    v12 = v4 - 1;
    v13 = *a2;
    v14 = 1;
    v15 = 0;
    v16 = v12 & v11;
    v17 = (_QWORD *)(v7 + 8LL * v16);
    v18 = *v17;
    if ( *v17 == *a2 )
    {
LABEL_19:
      *a3 = v17;
      return 1;
    }
    else
    {
      while ( v18 != -4096 )
      {
        if ( v18 != -8192 || v15 )
          v17 = v15;
        v16 = v12 & (v14 + v16);
        v18 = *(_QWORD *)(v7 + 8LL * v16);
        if ( v18 == v13 )
        {
          v17 = (_QWORD *)(v7 + 8LL * v16);
          goto LABEL_19;
        }
        ++v14;
        v15 = v17;
        v17 = (_QWORD *)(v7 + 8LL * v16);
      }
      if ( !v15 )
        v15 = v17;
      *a3 = v15;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}

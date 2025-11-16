// Function: sub_1E69FD0
// Address: 0x1e69fd0
//
__int64 __fastcall sub_1E69FD0(_QWORD *a1, unsigned int a2)
{
  __int64 (*v3)(); // rax
  _QWORD *v4; // r14
  __int64 (*v5)(); // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rcx
  __int64 v9; // r10
  unsigned int v10; // edx
  __int16 v11; // r12
  _WORD *v12; // rdx
  _WORD *v13; // rcx
  __int64 v14; // rsi
  unsigned __int16 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 result; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 (*v25)(); // rax
  __int16 v26; // ax
  unsigned int v27; // [rsp+0h] [rbp-70h] BYREF
  _QWORD *v28; // [rsp+8h] [rbp-68h]
  int v29; // [rsp+10h] [rbp-60h]
  unsigned __int16 v30; // [rsp+18h] [rbp-58h]
  _WORD *v31; // [rsp+20h] [rbp-50h]
  int v32; // [rsp+28h] [rbp-48h]
  unsigned __int16 v33; // [rsp+30h] [rbp-40h]
  __int64 v34; // [rsp+38h] [rbp-38h]

  v3 = *(__int64 (**)())(**(_QWORD **)(*a1 + 16LL) + 112LL);
  if ( v3 == sub_1D00B10 )
    BUG();
  v4 = (_QWORD *)v3();
  v5 = *(__int64 (**)())(*v4 + 72LL);
  if ( v5 == sub_1E693A0 || (result = ((__int64 (__fastcall *)(_QWORD *, _QWORD))v5)(v4, a2), !(_BYTE)result) )
  {
    v6 = 0;
    v7 = 0;
    LOBYTE(v29) = 1;
    v30 = 0;
    v31 = 0;
    v34 = 0;
    v8 = v4[7];
    v28 = v4 + 1;
    v32 = 0;
    v33 = 0;
    v9 = v4[1];
    v27 = a2;
    v10 = *(_DWORD *)(v9 + 24LL * a2 + 16);
    v11 = (v10 & 0xF) * a2;
    v12 = (_WORD *)(v8 + 2LL * (v10 >> 4));
    v13 = v12 + 1;
    v30 = *v12 + v11;
    v31 = v12 + 1;
    while ( 1 )
    {
      if ( !v13 )
        return 1;
      v14 = v30;
      v32 = *(_DWORD *)(v4[6] + 4LL * v30);
      v15 = v32;
      if ( (_WORD)v32 )
        break;
LABEL_20:
      v31 = ++v13;
      v26 = *(v13 - 1);
      v30 += v26;
      if ( !v26 )
        return 1;
    }
    while ( 1 )
    {
      v16 = v15;
      v17 = *(unsigned int *)(v4[1] + 24LL * v15 + 8);
      v18 = v4[7];
      v33 = v15;
      v34 = v18 + 2 * v17;
      if ( v34 )
        break;
      v15 = HIWORD(v32);
      v32 = HIWORD(v32);
      if ( !v15 )
        goto LABEL_20;
    }
    while ( 1 )
    {
      v19 = a1[34];
      v20 = 8 * v16;
      v21 = *(_QWORD *)(v19 + 8 * v16);
      if ( v21 )
      {
        if ( (*(_BYTE *)(v21 + 3) & 0x10) != 0 )
          break;
        v23 = *(_QWORD *)(v21 + 32);
        if ( v23 )
        {
          if ( (*(_BYTE *)(v23 + 3) & 0x10) != 0 )
            break;
        }
      }
      v24 = *(_QWORD *)(*a1 + 16LL);
      v25 = *(__int64 (**)())(*(_QWORD *)v24 + 112LL);
      if ( v25 == sub_1D00B10 )
        BUG();
      if ( *(_BYTE *)(*(_QWORD *)(((__int64 (__fastcall *)(__int64, __int64, __int64, _WORD *, __int64, __int64, unsigned int, _QWORD *, int))v25)(
                                    v24,
                                    v14,
                                    v19,
                                    v13,
                                    v6,
                                    v7,
                                    v27,
                                    v28,
                                    v29)
                                + 232)
                    + v20
                    + 4) )
      {
        v14 = a1[38];
        if ( (*(_QWORD *)(v14 + 8 * ((unsigned __int64)v15 >> 6)) & (1LL << v15)) == 0 )
          break;
      }
      sub_1E1D5E0((__int64)&v27);
      if ( !v31 )
        return 1;
      v16 = v33;
      v15 = v33;
    }
    return 0;
  }
  return result;
}

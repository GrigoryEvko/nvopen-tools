// Function: sub_A7FD90
// Address: 0xa7fd90
//
__int64 __fastcall sub_A7FD90(unsigned int **a1, __int64 a2, unsigned int a3)
{
  unsigned int **v4; // r12
  __int64 v5; // r14
  unsigned int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r9
  _BYTE *v10; // r10
  unsigned int v11; // esi
  unsigned int v12; // eax
  unsigned int v13; // edx
  __int64 v14; // rcx
  unsigned int *v15; // rdi
  __int64 (__fastcall *v16)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v17; // rax
  _BYTE *v18; // r15
  __int64 v20; // rax
  __int64 v21; // rdi
  unsigned int *v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rax
  _BYTE *v26; // [rsp+0h] [rbp-1A0h]
  __int64 v27; // [rsp+8h] [rbp-198h]
  int v28; // [rsp+8h] [rbp-198h]
  int v29; // [rsp+8h] [rbp-198h]
  unsigned int **v30; // [rsp+8h] [rbp-198h]
  int v31; // [rsp+8h] [rbp-198h]
  char v32[32]; // [rsp+10h] [rbp-190h] BYREF
  __int16 v33; // [rsp+30h] [rbp-170h]
  unsigned int v34; // [rsp+40h] [rbp-160h] BYREF
  __int16 v35; // [rsp+60h] [rbp-140h]
  _QWORD v36[4]; // [rsp+70h] [rbp-130h] BYREF
  char v37; // [rsp+90h] [rbp-110h]
  char v38; // [rsp+91h] [rbp-10Fh]

  v4 = a1;
  v5 = *(_QWORD *)(a2 + 8);
  v6 = 8 * *(_DWORD *)(v5 + 32);
  v7 = sub_BCB2B0(a1[9]);
  v8 = sub_BCDA70(v7, v6);
  v36[0] = "cast";
  v27 = v8;
  v38 = 1;
  v37 = 3;
  v26 = (_BYTE *)sub_A7EAA0(a1, 0x31u, a2, v8, (__int64)v36, 0, v34, 0);
  v10 = (_BYTE *)sub_AD6530(v27);
  if ( a3 <= 0xF )
  {
    if ( v6 )
    {
      v11 = a3 - v6;
      v9 = v6 - a3 + 16;
      do
      {
        v12 = v6 - a3;
        do
        {
          v13 = 16 - v6 + v12;
          v14 = v11 + v12;
          if ( v6 <= v12 )
            v13 = v12;
          ++v12;
          *((_DWORD *)v36 + v14) = v6 - a3 + v11 + v13;
        }
        while ( v12 != (_DWORD)v9 );
        v11 += 16;
      }
      while ( a3 != v11 );
    }
    v15 = a1[10];
    v33 = 257;
    v16 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v15 + 112LL);
    if ( v16 == sub_9B6630 )
    {
      if ( *v10 > 0x15u || *v26 > 0x15u )
        goto LABEL_16;
      v28 = (int)v10;
      v17 = sub_AD5CE0(v10, v26, v36, v6, 0, v9);
      LODWORD(v10) = v28;
      v18 = (_BYTE *)v17;
    }
    else
    {
      v31 = (int)v10;
      v25 = ((__int64 (__fastcall *)(unsigned int *, _BYTE *, _BYTE *, _QWORD *, _QWORD))v16)(v15, v10, v26, v36, v6);
      LODWORD(v10) = v31;
      v18 = (_BYTE *)v25;
    }
    if ( v18 )
    {
LABEL_14:
      v10 = v18;
      goto LABEL_15;
    }
LABEL_16:
    v29 = (int)v10;
    v35 = 257;
    v20 = sub_BD2C40(112, unk_3F1FE60);
    v18 = (_BYTE *)v20;
    if ( v20 )
      sub_B4E9E0(v20, v29, (_DWORD)v26, (unsigned int)v36, v6, (unsigned int)&v34, 0, 0);
    (*(void (__fastcall **)(unsigned int *, _BYTE *, char *, unsigned int *, unsigned int *))(*(_QWORD *)v4[11] + 16LL))(
      v4[11],
      v18,
      v32,
      v4[7],
      v4[8]);
    v21 = (__int64)&(*v4)[4 * *((unsigned int *)v4 + 2)];
    if ( *v4 != (unsigned int *)v21 )
    {
      v30 = v4;
      v22 = *v4;
      do
      {
        v23 = *((_QWORD *)v22 + 1);
        v24 = *v22;
        v22 += 4;
        sub_B99FD0(v18, v24, v23);
      }
      while ( (unsigned int *)v21 != v22 );
      v4 = v30;
    }
    goto LABEL_14;
  }
LABEL_15:
  v36[0] = "cast";
  v38 = 1;
  v37 = 3;
  return sub_A7EAA0(v4, 0x31u, (__int64)v10, v5, (__int64)v36, 0, v34, 0);
}

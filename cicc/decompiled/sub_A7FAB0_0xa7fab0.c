// Function: sub_A7FAB0
// Address: 0xa7fab0
//
__int64 __fastcall sub_A7FAB0(unsigned int **a1, __int64 a2, unsigned int a3)
{
  unsigned int **v3; // r12
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r9
  _BYTE *v11; // r10
  int v12; // r11d
  unsigned int v13; // eax
  unsigned int v14; // edx
  unsigned int *v15; // rdi
  __int64 (__fastcall *v16)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v17; // rax
  _BYTE *v18; // rbx
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned int *v22; // r12
  unsigned int *v23; // r15
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rax
  _BYTE *v27; // [rsp+10h] [rbp-1A0h]
  __int64 v28; // [rsp+18h] [rbp-198h]
  _BYTE *v29; // [rsp+18h] [rbp-198h]
  int v30; // [rsp+18h] [rbp-198h]
  int v31; // [rsp+18h] [rbp-198h]
  unsigned int **v32; // [rsp+18h] [rbp-198h]
  int v33; // [rsp+18h] [rbp-198h]
  char v34[32]; // [rsp+20h] [rbp-190h] BYREF
  __int16 v35; // [rsp+40h] [rbp-170h]
  unsigned int v36; // [rsp+50h] [rbp-160h] BYREF
  __int16 v37; // [rsp+70h] [rbp-140h]
  _QWORD v38[4]; // [rsp+80h] [rbp-130h] BYREF
  char v39; // [rsp+A0h] [rbp-110h]
  char v40; // [rsp+A1h] [rbp-10Fh]

  v3 = a1;
  v5 = *(_QWORD *)(a2 + 8);
  v6 = (unsigned int)(8 * *(_DWORD *)(v5 + 32));
  v7 = sub_BCB2B0(a1[9]);
  v8 = sub_BCDA70(v7, (unsigned int)v6);
  v38[0] = "cast";
  v28 = v8;
  v40 = 1;
  v39 = 3;
  v27 = (_BYTE *)sub_A7EAA0(a1, 0x31u, a2, v8, (__int64)v38, 0, v36, 0);
  v9 = sub_AD6530(v28);
  v11 = (_BYTE *)v9;
  if ( a3 <= 0xF )
  {
    if ( (_DWORD)v6 )
    {
      v29 = (_BYTE *)v9;
      v12 = -a3;
      v10 = (unsigned int)(v6 - 16);
      do
      {
        v13 = a3 + 1;
        v14 = a3;
        while ( 1 )
        {
          *((_DWORD *)v38 + v12 + v13 - 1) = a3 + v12 + v14;
          if ( a3 + 16 == v13 )
            break;
          v14 = v10 + v13;
          if ( v13 <= 0xF )
            v14 = v13;
          ++v13;
        }
        v12 += 16;
      }
      while ( (_DWORD)v6 - a3 != v12 );
      v11 = v29;
    }
    v15 = a1[10];
    v35 = 257;
    v16 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v15 + 112LL);
    if ( v16 == sub_9B6630 )
    {
      if ( *v27 > 0x15u || *v11 > 0x15u )
        goto LABEL_18;
      v30 = (int)v11;
      v17 = sub_AD5CE0(v27, v11, v38, v6, 0, v10);
      LODWORD(v11) = v30;
      v18 = (_BYTE *)v17;
    }
    else
    {
      v33 = (int)v11;
      v26 = ((__int64 (__fastcall *)(unsigned int *, _BYTE *, _BYTE *, _QWORD *, __int64))v16)(v15, v27, v11, v38, v6);
      LODWORD(v11) = v33;
      v18 = (_BYTE *)v26;
    }
    if ( v18 )
    {
LABEL_16:
      v11 = v18;
      goto LABEL_17;
    }
LABEL_18:
    v31 = (int)v11;
    v37 = 257;
    v20 = sub_BD2C40(112, unk_3F1FE60);
    v18 = (_BYTE *)v20;
    if ( v20 )
      sub_B4E9E0(v20, (_DWORD)v27, v31, (unsigned int)v38, v6, (unsigned int)&v36, 0, 0);
    (*(void (__fastcall **)(unsigned int *, _BYTE *, char *, unsigned int *, unsigned int *))(*(_QWORD *)v3[11] + 16LL))(
      v3[11],
      v18,
      v34,
      v3[7],
      v3[8]);
    v21 = (__int64)&(*v3)[4 * *((unsigned int *)v3 + 2)];
    if ( *v3 != (unsigned int *)v21 )
    {
      v32 = v3;
      v22 = *v3;
      v23 = (unsigned int *)v21;
      do
      {
        v24 = *((_QWORD *)v22 + 1);
        v25 = *v22;
        v22 += 4;
        sub_B99FD0(v18, v25, v24);
      }
      while ( v23 != v22 );
      v3 = v32;
    }
    goto LABEL_16;
  }
LABEL_17:
  v38[0] = "cast";
  v40 = 1;
  v39 = 3;
  return sub_A7EAA0(v3, 0x31u, (__int64)v11, v5, (__int64)v38, 0, v36, 0);
}

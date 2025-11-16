// Function: sub_1D03960
// Address: 0x1d03960
//
__int64 __fastcall sub_1D03960(_QWORD *a1)
{
  __int64 *v1; // rax
  __int64 v2; // r13
  __int64 *v3; // rbx
  __int64 v5; // r15
  __int64 v6; // r14
  unsigned __int8 v7; // dl
  unsigned __int8 v8; // al
  int v9; // eax
  bool v10; // cf
  bool v11; // al
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rcx
  char v15; // r9
  __int64 v16; // rdi
  __int64 (*v17)(); // rax
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 (*v20)(); // rax
  int v21; // eax
  unsigned int v22; // r14d
  __int64 result; // rax
  bool v24; // zf
  bool v25; // si
  bool v26; // al
  int v27; // eax
  int v28; // eax
  unsigned int v29; // r14d
  int v30; // eax
  int v31; // eax
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  char v34; // [rsp+13h] [rbp-4Dh]
  int v35; // [rsp+14h] [rbp-4Ch]
  char v36; // [rsp+14h] [rbp-4Ch]
  unsigned __int8 v37; // [rsp+14h] [rbp-4Ch]
  int v38; // [rsp+14h] [rbp-4Ch]
  int v39; // [rsp+14h] [rbp-4Ch]
  char v40; // [rsp+14h] [rbp-4Ch]
  int v41; // [rsp+14h] [rbp-4Ch]
  char v42; // [rsp+14h] [rbp-4Ch]
  __int64 *v43; // [rsp+18h] [rbp-48h]
  unsigned int v44; // [rsp+28h] [rbp-38h] BYREF
  unsigned int v45[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v1 = (__int64 *)a1[3];
  v2 = a1[2];
  v43 = v1;
  if ( v1 == (__int64 *)v2 )
    return 0;
  v3 = (__int64 *)(v2 + 8);
  if ( v1 == (__int64 *)(v2 + 8) )
    goto LABEL_39;
  do
  {
    v5 = *v3;
    v6 = *(_QWORD *)v2;
    v7 = (*(_BYTE *)(*(_QWORD *)v2 + 229LL) & 0x10) != 0;
    v8 = (*(_BYTE *)(*v3 + 229) & 0x10) != 0;
    if ( v7 == v8 )
    {
      if ( (*(_BYTE *)(v6 + 228) & 2) == 0 && (*(_BYTE *)(v5 + 228) & 2) == 0 )
      {
        v44 = 0;
        v45[0] = 0;
        if ( !byte_4FC12C0 || !byte_4FC11E0 )
        {
          v35 = sub_1D01500((_QWORD *)a1[21], (__int64 *)v6, &v44);
          v9 = sub_1D01500((_QWORD *)a1[21], (__int64 *)v5, v45);
          if ( !byte_4FC12C0 )
          {
            v24 = v35 == v9;
            v11 = v35 > v9;
            if ( !v24 )
            {
LABEL_44:
              if ( v11 )
                v2 = (__int64)v3;
              goto LABEL_37;
            }
            if ( v35 > 0 )
            {
              v25 = sub_1D00FE0((_DWORD *)v6);
              v26 = sub_1D00FE0((_DWORD *)v5);
              if ( !v26 && v25 )
                goto LABEL_37;
              if ( !v25 && v26 )
              {
                v2 = (__int64)v3;
                goto LABEL_37;
              }
            }
          }
          if ( !byte_4FC11E0 )
          {
            v10 = v44 < v45[0];
            if ( v44 != v45[0] )
            {
LABEL_11:
              v11 = v10;
              goto LABEL_44;
            }
          }
        }
        v12 = (unsigned __int8)byte_4FC0F40;
        if ( !byte_4FC0F40 )
        {
          v13 = a1[21];
          if ( (*(_BYTE *)(v6 + 236) & 2) == 0 )
          {
            v33 = a1[21];
            v37 = byte_4FC0F40;
            sub_1F01F70(v6);
            v13 = v33;
            v12 = v37;
          }
          v14 = *(unsigned int *)(v13 + 8);
          v15 = 1;
          if ( *(_DWORD *)(v6 + 244) <= (int)v14 )
          {
            v15 = 0;
            v16 = *(_QWORD *)(*(_QWORD *)(v13 + 88) + 704LL);
            v17 = *(__int64 (**)())(*(_QWORD *)v16 + 24LL);
            if ( v17 != sub_1D00B90 )
            {
              v40 = v12;
              v30 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64, __int64, _QWORD))v17)(
                      v16,
                      v6,
                      0,
                      v14,
                      v12,
                      0);
              LOBYTE(v12) = v40;
              v15 = v30 != 0;
            }
          }
          v18 = a1[21];
          if ( (*(_BYTE *)(v5 + 236) & 2) == 0 )
          {
            v34 = v15;
            v32 = a1[21];
            v36 = v12;
            sub_1F01F70(v5);
            v15 = v34;
            v18 = v32;
            LOBYTE(v12) = v36;
          }
          if ( *(_DWORD *)(v5 + 244) > *(_DWORD *)(v18 + 8) )
          {
            LOBYTE(v12) = 1;
          }
          else
          {
            v19 = *(_QWORD *)(*(_QWORD *)(v18 + 88) + 704LL);
            v20 = *(__int64 (**)())(*(_QWORD *)v19 + 24LL);
            if ( v20 != sub_1D00B90 )
            {
              v42 = v15;
              v31 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v20)(v19, v5, 0);
              v15 = v42;
              LOBYTE(v12) = v31 != 0;
            }
          }
          if ( (_BYTE)v12 != v15 )
            goto LABEL_64;
        }
        if ( !byte_4FC0E60 )
        {
          if ( (*(_BYTE *)(v6 + 236) & 1) == 0 )
            sub_1F01DD0(v6);
          v21 = *(_DWORD *)(v6 + 240);
          if ( (*(_BYTE *)(v5 + 236) & 1) == 0 )
          {
            v38 = *(_DWORD *)(v6 + 240);
            sub_1F01DD0(v5);
            v21 = v38;
          }
          if ( (int)abs32(v21 - *(_DWORD *)(v5 + 240)) > dword_4FC0BC0 )
          {
            if ( (*(_BYTE *)(v6 + 236) & 1) == 0 )
              sub_1F01DD0(v6);
            v22 = *(_DWORD *)(v6 + 240);
            if ( (*(_BYTE *)(v5 + 236) & 1) == 0 )
              sub_1F01DD0(v5);
            v10 = v22 < *(_DWORD *)(v5 + 240);
            goto LABEL_11;
          }
        }
        if ( !byte_4FC0D80 )
        {
          if ( (*(_BYTE *)(v6 + 236) & 2) == 0 )
            sub_1F01F70(v6);
          v27 = *(_DWORD *)(v6 + 244);
          if ( (*(_BYTE *)(v5 + 236) & 2) == 0 )
          {
            v39 = *(_DWORD *)(v6 + 244);
            sub_1F01F70(v5);
            v27 = v39;
          }
          if ( v27 != *(_DWORD *)(v5 + 244) )
          {
            if ( (*(_BYTE *)(v6 + 236) & 2) == 0 )
              sub_1F01F70(v6);
            v28 = *(_DWORD *)(v6 + 244);
            if ( (*(_BYTE *)(v5 + 236) & 2) == 0 )
            {
              v41 = *(_DWORD *)(v6 + 244);
              sub_1F01F70(v5);
              v28 = v41;
            }
            if ( (int)abs32(v28 - *(_DWORD *)(v5 + 244)) > dword_4FC0BC0 )
            {
LABEL_64:
              if ( (*(_BYTE *)(v6 + 236) & 2) == 0 )
                sub_1F01F70(v6);
              v29 = *(_DWORD *)(v6 + 244);
              if ( (*(_BYTE *)(v5 + 236) & 2) == 0 )
                sub_1F01F70(v5);
              v11 = v29 > *(_DWORD *)(v5 + 244);
              goto LABEL_44;
            }
          }
        }
      }
      v11 = sub_1D03130(v6, v5, a1[21]);
      goto LABEL_44;
    }
    if ( v7 < v8 )
      v2 = (__int64)v3;
LABEL_37:
    ++v3;
  }
  while ( v43 != v3 );
  v3 = (__int64 *)a1[3];
LABEL_39:
  result = *(_QWORD *)v2;
  if ( (__int64 *)v2 != v3 - 1 )
  {
    *(_QWORD *)v2 = *(v3 - 1);
    *(v3 - 1) = result;
    v2 = a1[3] - 8LL;
  }
  a1[3] = v2;
  *(_DWORD *)(result + 196) = 0;
  return result;
}

// Function: sub_1920970
// Address: 0x1920970
//
void __fastcall sub_1920970(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  signed __int64 v5; // rbx
  char *v6; // r9
  char *v7; // r10
  signed __int64 v8; // r12
  __int64 v9; // r13
  char *v10; // rax
  char *v11; // r10
  char *v12; // r11
  char *v13; // r9
  char *v14; // r15
  __int64 v15; // r14
  unsigned __int64 v16; // rax
  char *v17; // rax
  int v18; // ecx
  int v19; // edx
  unsigned int v20; // esi
  __int64 v21; // r8
  __int64 v22; // rdi
  unsigned __int64 v24; // [rsp+8h] [rbp-48h]
  char *v25; // [rsp+10h] [rbp-40h]
  char *v26; // [rsp+18h] [rbp-38h]
  char *v27; // [rsp+18h] [rbp-38h]
  char *v28; // [rsp+18h] [rbp-38h]

  if ( a5 )
  {
    v5 = a4;
    if ( a4 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a5;
      if ( a4 + a5 == 2 )
      {
        v16 = (unsigned __int64)a1;
        v14 = a2;
LABEL_12:
        v18 = *(_DWORD *)v14;
        v19 = *(_DWORD *)v16;
        if ( *(_DWORD *)v14 >= *(_DWORD *)v16 )
        {
          if ( v18 != v19 )
            return;
          v20 = *(_DWORD *)(v16 + 4);
          if ( *((_DWORD *)v14 + 1) >= v20 )
            return;
        }
        else
        {
          v20 = *(_DWORD *)(v16 + 4);
        }
        v21 = *(_QWORD *)(v16 + 8);
        v22 = *(_QWORD *)(v16 + 16);
        *(_DWORD *)v16 = v18;
        *(_DWORD *)(v16 + 4) = *((_DWORD *)v14 + 1);
        *(_QWORD *)(v16 + 8) = *((_QWORD *)v14 + 1);
        *(_QWORD *)(v16 + 16) = *((_QWORD *)v14 + 2);
        *(_DWORD *)v14 = v19;
        *((_DWORD *)v14 + 1) = v20;
        *((_QWORD *)v14 + 1) = v21;
        *((_QWORD *)v14 + 2) = v22;
        return;
      }
      if ( a4 <= a5 )
        goto LABEL_10;
LABEL_5:
      v26 = v6;
      v9 = v5 / 2;
      v10 = (char *)sub_1920010(
                      v7,
                      a3,
                      (unsigned int *)&v6[8 * (v5 / 2)
                                        + 8 * ((v5 + ((unsigned __int64)v5 >> 63)) & 0xFFFFFFFFFFFFFFFELL)]);
      v13 = v26;
      v14 = v10;
      v15 = 0xAAAAAAAAAAAAAAABLL * ((v10 - v11) >> 3);
      while ( 1 )
      {
        v25 = v13;
        v27 = v12;
        v8 -= v15;
        v24 = sub_191FBD0(v12, v11, v14);
        sub_1920970(v25, v27, v24, v9, v15);
        v5 -= v9;
        if ( !v5 )
          break;
        v16 = v24;
        if ( !v8 )
          break;
        if ( v8 + v5 == 2 )
          goto LABEL_12;
        v6 = (char *)v24;
        v7 = v14;
        if ( v5 > v8 )
          goto LABEL_5;
LABEL_10:
        v28 = v6;
        v15 = v8 / 2;
        v14 = &v7[8 * (v8 / 2) + 8 * ((v8 + ((unsigned __int64)v8 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
        v17 = (char *)sub_1920080(v6, (__int64)v7, (unsigned int *)v14);
        v13 = v28;
        v12 = v17;
        v9 = 0xAAAAAAAAAAAAAAABLL * ((v17 - v28) >> 3);
      }
    }
  }
}

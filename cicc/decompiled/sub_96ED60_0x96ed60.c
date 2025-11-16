// Function: sub_96ED60
// Address: 0x96ed60
//
__int64 __fastcall sub_96ED60(__int64 a1, __int64 a2, char a3)
{
  char v5; // al
  __int64 v7; // r15
  _BYTE *v8; // rax
  int v9; // eax
  __int64 v10; // r15
  int v11; // ebx
  int v12; // ebx
  unsigned int v13; // r14d
  __int64 v14; // rsi
  unsigned __int8 *v15; // r15
  int v16; // edx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  _BYTE *v19; // rdi
  unsigned int v20; // ebx
  __int64 v21; // r12
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  char v25; // al
  __int16 v26; // ax
  char v27; // dl
  _QWORD *i; // r13
  int v30; // [rsp+28h] [rbp-E8h]
  __int64 v32; // [rsp+30h] [rbp-E0h] BYREF
  _QWORD *v33; // [rsp+38h] [rbp-D8h]
  _BYTE *v34; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v35; // [rsp+58h] [rbp-B8h]
  _BYTE v36[176]; // [rsp+60h] [rbp-B0h] BYREF

  v5 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 18 )
    return sub_96AEE0(a1, a2, a3);
  if ( (unsigned __int8)(v5 - 12) > 2u && v5 != 5 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    {
      v8 = (_BYTE *)sub_AD7630(a1, 0);
      if ( v8 && *v8 == 18 )
      {
        if ( sub_96AEE0((__int64)v8, a2, a3) )
        {
          v9 = *(_DWORD *)(v7 + 32);
          BYTE4(v34) = *(_BYTE *)(v7 + 8) == 18;
          LODWORD(v34) = v9;
          return sub_AD5E10((size_t)v34);
        }
        return 0;
      }
      v7 = *(_QWORD *)(v7 + 24);
      v5 = *(_BYTE *)a1;
    }
    if ( v5 == 11 )
    {
      v11 = *(_DWORD *)(a1 + 4);
      v34 = v36;
      v35 = 0x1000000000LL;
      v12 = v11 & 0x7FFFFFF;
      if ( v12 )
      {
        v13 = 0;
        while ( 1 )
        {
          v14 = v13;
          v15 = (unsigned __int8 *)sub_AD69F0(a1, v13);
          v16 = *v15;
          if ( (unsigned int)(v16 - 12) > 1 )
          {
            if ( (_BYTE)v16 != 18 )
              break;
            v14 = a2;
            v15 = (unsigned __int8 *)sub_96AEE0((__int64)v15, a2, a3);
            if ( !v15 )
              break;
          }
          v17 = (unsigned int)v35;
          v18 = (unsigned int)v35 + 1LL;
          if ( v18 > HIDWORD(v35) )
          {
            sub_C8D5F0(&v34, v36, v18, 8);
            v17 = (unsigned int)v35;
          }
          ++v13;
          *(_QWORD *)&v34[8 * v17] = v15;
          LODWORD(v35) = v35 + 1;
          if ( v13 == v12 )
          {
            v19 = v34;
            v14 = (unsigned int)v35;
            goto LABEL_21;
          }
        }
        v10 = 0;
        goto LABEL_25;
      }
      v19 = v36;
      v14 = 0;
    }
    else
    {
      if ( v5 != 16 )
        return 0;
      v34 = v36;
      v35 = 0x1000000000LL;
      v30 = sub_AC5290(a1);
      if ( v30 )
      {
        v20 = 0;
        v21 = sub_C33340();
        sub_AC5470(&v32, a1, 0);
        while ( 1 )
        {
          if ( v32 == v21 )
            v25 = sub_C40310(&v32);
          else
            v25 = sub_C33940(&v32);
          if ( v25 )
          {
            v26 = sub_968EE0(a2, v7);
            v27 = v26;
            if ( !a3 )
              v27 = HIBYTE(v26);
            v14 = (__int64)&v32;
            v22 = sub_96AC80((_QWORD *)v7, &v32, v27);
            if ( !v22 )
            {
              v10 = 0;
              sub_91D830(&v32);
              goto LABEL_25;
            }
          }
          else
          {
            v22 = sub_AD8F10(v7, &v32);
          }
          v23 = (unsigned int)v35;
          v24 = (unsigned int)v35 + 1LL;
          if ( v24 > HIDWORD(v35) )
          {
            sub_C8D5F0(&v34, v36, v24, 8);
            v23 = (unsigned int)v35;
          }
          *(_QWORD *)&v34[8 * v23] = v22;
          LODWORD(v35) = v35 + 1;
          if ( v32 == v21 )
          {
            if ( v33 )
            {
              for ( i = &v33[3 * *(v33 - 1)]; v33 != i; sub_969EE0((__int64)i) )
              {
                while ( 1 )
                {
                  i -= 3;
                  if ( v21 == *i )
                    break;
                  sub_C338F0(i);
                  if ( v33 == i )
                    goto LABEL_53;
                }
              }
LABEL_53:
              j_j_j___libc_free_0_0(i - 1);
            }
          }
          else
          {
            sub_C338F0(&v32);
          }
          if ( v30 == ++v20 )
            break;
          sub_AC5470(&v32, a1, v20);
        }
      }
      v14 = (unsigned int)v35;
      v19 = v34;
    }
LABEL_21:
    v10 = sub_AD3730(v19, v14);
LABEL_25:
    if ( v34 != v36 )
      _libc_free(v34, v14);
    return v10;
  }
  return a1;
}

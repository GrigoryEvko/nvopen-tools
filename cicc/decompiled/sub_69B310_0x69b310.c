// Function: sub_69B310
// Address: 0x69b310
//
__int64 __fastcall sub_69B310(__int64 *a1, __int64 *a2, unsigned __int16 a3, _DWORD *a4, int a5, __int64 a6)
{
  __int64 v10; // r14
  __int64 v11; // rdi
  bool v12; // zf
  char v13; // dl
  __int64 v14; // rax
  __int64 v15; // rdi
  int v16; // r14d
  int v17; // edx
  __int64 v18; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  char i; // dl
  __int64 *v23; // rdx
  int v24; // eax
  unsigned int v25; // eax
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 j; // r8
  char v39; // dl
  __int64 k; // r14
  char v41; // si
  __int64 m; // rax
  __int64 v43; // r14
  __int64 v44; // rdi
  int v45; // [rsp+0h] [rbp-70h]
  __int64 v46; // [rsp+0h] [rbp-70h]
  unsigned int v47; // [rsp+8h] [rbp-68h]
  __int64 v48; // [rsp+8h] [rbp-68h]
  int v49; // [rsp+10h] [rbp-60h]
  __int64 *v50; // [rsp+10h] [rbp-60h]
  unsigned __int8 v53; // [rsp+2Fh] [rbp-41h] BYREF
  int v54; // [rsp+30h] [rbp-40h] BYREF
  int v55; // [rsp+34h] [rbp-3Ch] BYREF
  _QWORD v56[7]; // [rsp+38h] [rbp-38h] BYREF

  v54 = 0;
  if ( dword_4F077C4 != 2 )
    goto LABEL_101;
  if ( (unsigned int)sub_68FE10(a1, 1, 1) || (unsigned int)sub_68FE10(a2, 0, 1) )
    sub_84EC30(byte_4B6D300[a3], 0, 0, 1, 0, (_DWORD)a1, (__int64)a2, (__int64)a4, a5, 0, 0, a6, 0, 0, (__int64)&v54);
  if ( !v54 )
  {
LABEL_101:
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) != 2 || (sub_68BB70(a1, a2, a4, a6, &v54), !v54) )
    {
      sub_6F69D0(a1, 0);
      LODWORD(v10) = sub_8D2D50(*a1);
      if ( (_DWORD)v10 || HIDWORD(qword_4F077B4) && (unsigned int)sub_8D2B80(*a1) )
      {
        v45 = 0;
        LODWORD(v10) = 0;
      }
      else
      {
        v45 = sub_8D2660(*a1);
        if ( v45 )
        {
          v45 = 1;
        }
        else
        {
          v25 = sub_6E9530();
          v10 = (unsigned int)sub_6FB4D0(a1, v25) != 0;
        }
      }
      sub_6F69D0(a2, 0);
      v11 = *a1;
      v12 = *((_BYTE *)a1 + 16) == 0;
      v56[0] = *a1;
      if ( !v12 )
      {
        v13 = *(_BYTE *)(v11 + 140);
        if ( v13 == 12 )
        {
          v14 = v11;
          do
          {
            v14 = *(_QWORD *)(v14 + 160);
            v13 = *(_BYTE *)(v14 + 140);
          }
          while ( v13 == 12 );
        }
        if ( v13 && *((_BYTE *)a2 + 16) )
        {
          v21 = *a2;
          for ( i = *(_BYTE *)(*a2 + 140); i == 12; i = *(_BYTE *)(v21 + 140) )
            v21 = *(_QWORD *)(v21 + 160);
          if ( i )
          {
            if ( (unsigned int)sub_8D28B0(v11) || (v49 = sub_8D28B0(*a2)) != 0 )
            {
              sub_6E8B30(a1, a2, v56);
              v49 = 0;
              v15 = v56[0];
              v47 = a3;
              goto LABEL_11;
            }
            v26 = *a1;
            if ( (unsigned int)sub_8D2AF0(*a1) )
            {
              if ( (unsigned int)sub_6E5430(v26, 0, v27, v28, v29, v30) )
              {
                v26 = 1044;
                sub_6851C0(0x414u, (_DWORD *)a1 + 17);
              }
              v56[0] = sub_72C930(v26);
              v15 = v56[0];
              v47 = a3;
LABEL_11:
              if ( (unsigned int)sub_8D2B80(v15) )
              {
                v20 = sub_6E8E20(v56[0]);
                v16 = v20;
                if ( *(_BYTE *)(v20 + 140) == 15 )
                  *(_BYTE *)(v20 + 176) |= 1u;
              }
              else
              {
                v16 = sub_6EFF80();
              }
              v53 = sub_6E9930(v47, v56[0]);
              sub_6FC7D0(v56[0], a1, a2, v53);
              if ( !v49 || *(char *)(qword_4D03C50 + 21LL) < 0 )
                goto LABEL_15;
              v23 = a1;
              if ( v55 )
                v23 = a2;
              if ( *((_BYTE *)v23 + 16) != 2 )
                goto LABEL_15;
              v50 = v23;
              v48 = (__int64)(v23 + 18);
              if ( !(unsigned int)sub_8D2930(v23[34]) || *((_BYTE *)v50 + 317) != 1 )
                goto LABEL_15;
              v24 = sub_6210B0(v48, 0);
              if ( v24 )
              {
                if ( v24 < 0 && (unsigned int)sub_6E53E0(5, 514, a4) )
                  sub_684B30(0x202u, a4);
                goto LABEL_15;
              }
              if ( v55 )
              {
                if ( a3 == 46 || a3 == 43 )
                {
LABEL_47:
                  if ( (unsigned int)sub_6E53E0(5, 186, a4) )
                    sub_684B30(0xBAu, a4);
                }
              }
              else if ( (unsigned __int16)(a3 - 44) <= 1u )
              {
                goto LABEL_47;
              }
LABEL_15:
              sub_7016A0(v53, (_DWORD)a1, (_DWORD)a2, v16, a6, (_DWORD)a4, a5);
              goto LABEL_16;
            }
            v49 = sub_8D2AF0(*a2);
            if ( !v49 )
            {
              v47 = a3;
              if ( (_DWORD)v10 || (unsigned int)sub_8D2E30(*a2) )
              {
                sub_6EB6C0((_DWORD)a1, (_DWORD)a2, (_DWORD)a4, byte_4B6D300[a3], 0, 0, 1, 0, (__int64)v56);
                v15 = v56[0];
                goto LABEL_11;
              }
              if ( v45 || (v49 = sub_8D2660(*a2)) != 0 )
              {
                v49 = sub_8D3D10(*a2);
                if ( v49 )
                {
                  v43 = *a2;
                  v44 = (unsigned int)sub_6E9530();
                  sub_6E5E80(v44, (char *)a2 + 68, v43);
                  v49 = 0;
                  v56[0] = sub_72C930(v44);
                }
                else
                {
                  sub_6E8FF0(a1, a2, a4, v56);
                }
                v15 = v56[0];
                goto LABEL_11;
              }
              if ( HIDWORD(qword_4F077B4) && (unsigned int)sub_6FD310(a3, a1, a2, a4, v56, &v53) )
                goto LABEL_87;
              v49 = sub_6E9580(a2);
              if ( v49 )
                v49 = sub_68B1F0(a1, a2, &v55);
              v56[0] = sub_6E8B10(a1, a2, v35, v36, v37);
              if ( !(unsigned int)sub_8D2930(v56[0]) )
              {
LABEL_87:
                v15 = v56[0];
                goto LABEL_11;
              }
              for ( j = *a1; ; j = *(_QWORD *)(j + 160) )
              {
                v39 = *(_BYTE *)(j + 140);
                if ( v39 != 12 )
                  break;
              }
              for ( k = *a2; ; k = *(_QWORD *)(k + 160) )
              {
                v41 = *(_BYTE *)(k + 140);
                if ( v41 != 12 )
                  break;
              }
              v15 = v56[0];
              for ( m = v56[0]; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
                ;
              if ( !byte_4B6DF90[*(unsigned __int8 *)(m + 160)] )
              {
                if ( v39 != 2 || !byte_4B6DF90[*(unsigned __int8 *)(j + 160)] || *((_BYTE *)a1 + 16) == 2 )
                {
                  if ( v41 != 2 )
                    goto LABEL_11;
                  if ( !byte_4B6DF90[*(unsigned __int8 *)(k + 160)] || *((_BYTE *)a2 + 16) == 2 )
                  {
                    if ( v39 != 2 )
                      goto LABEL_11;
LABEL_86:
                    sub_68B0C0((_DWORD *)j, k, (__int64)a4, 4u);
                    goto LABEL_87;
                  }
                }
                v46 = j;
                sub_6E5C80(4, 1873, a4);
                j = v46;
              }
              if ( *(_BYTE *)(j + 140) != 2 || *(_BYTE *)(k + 140) != 2 )
                goto LABEL_87;
              goto LABEL_86;
            }
            v11 = 1044;
            sub_69A8C0(1044, (_DWORD *)a2 + 17, v31, v32, v33, v34);
          }
        }
      }
      v49 = 0;
      v56[0] = sub_72C930(v11);
      v15 = v56[0];
      v47 = a3;
      goto LABEL_11;
    }
  }
LABEL_16:
  v17 = *((_DWORD *)a1 + 17);
  *(_WORD *)(a6 + 72) = *((_WORD *)a1 + 36);
  *(_DWORD *)(a6 + 68) = v17;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a6 + 68);
  v18 = *(__int64 *)((char *)a2 + 76);
  *(_QWORD *)(a6 + 76) = v18;
  unk_4F061D8 = v18;
  return sub_6E3280(a6, a4);
}

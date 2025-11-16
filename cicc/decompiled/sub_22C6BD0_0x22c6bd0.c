// Function: sub_22C6BD0
// Address: 0x22c6bd0
//
void __fastcall sub_22C6BD0(unsigned __int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r15
  int v12; // edx
  int v13; // esi
  __int64 v14; // rdi
  unsigned int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r10
  __int64 v19; // r11
  __int64 v20; // rdi
  char v21; // al
  __int64 v22; // rax
  _QWORD *v23; // r10
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned __int64 v27; // rax
  unsigned __int8 *v28; // rax
  int v29; // ecx
  int v30; // r9d
  _QWORD *v31; // [rsp+10h] [rbp-B0h]
  _QWORD *v32; // [rsp+18h] [rbp-A8h]
  __int64 v33; // [rsp+20h] [rbp-A0h]
  __int64 v34; // [rsp+20h] [rbp-A0h]
  __int64 v35; // [rsp+28h] [rbp-98h]
  __int64 v36; // [rsp+28h] [rbp-98h]
  __int64 **v37; // [rsp+28h] [rbp-98h]
  __int64 v38; // [rsp+28h] [rbp-98h]
  unsigned __int8 v39[48]; // [rsp+30h] [rbp-90h] BYREF
  void *v40; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v41[4]; // [rsp+68h] [rbp-58h] BYREF
  char v42; // [rsp+88h] [rbp-38h]

  v9 = a4;
  if ( !a4 )
  {
    v9 = a2;
    if ( *(_BYTE *)a2 <= 0x1Cu )
      return;
  }
  v10 = *(_QWORD *)(a1 + 240);
  v11 = *(_QWORD *)(v9 + 40);
  if ( *(_BYTE *)(v10 + 192) )
  {
    v12 = *(_DWORD *)(v10 + 184);
    if ( v12 )
      goto LABEL_4;
  }
  else
  {
    v38 = *(_QWORD *)(a1 + 240);
    sub_CFDFC0(v38, a2, (__int64)a3, a4, a5, a6);
    v10 = v38;
    v12 = *(_DWORD *)(v38 + 184);
    if ( v12 )
    {
LABEL_4:
      v13 = v12 - 1;
      v14 = *(_QWORD *)(v10 + 168);
      v41[0] = 2;
      v41[1] = 0;
      v15 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v41[2] = -4096;
      v41[3] = 0;
      v16 = v14 + 88LL * v15;
      v17 = *(_QWORD *)(v16 + 24);
      if ( a2 == v17 )
      {
LABEL_5:
        v33 = v16;
        v35 = v10;
        v40 = &unk_49DB368;
        sub_D68D70(v41);
        if ( v33 != *(_QWORD *)(v35 + 168) + 88LL * *(unsigned int *)(v35 + 184) )
        {
          v18 = *(_QWORD *)(v33 + 40);
          v19 = v18 + 32LL * *(unsigned int *)(v33 + 48);
          if ( v19 != v18 )
          {
            do
            {
              while ( 1 )
              {
                v20 = *(_QWORD *)(v18 + 16);
                if ( v20 )
                {
                  if ( *(_QWORD *)(v20 + 40) == v11 )
                  {
                    v34 = v19;
                    v36 = v18;
                    v21 = sub_98CF40(v20, v9, 0, 0);
                    v18 = v36;
                    v19 = v34;
                    if ( v21 )
                    {
                      sub_22C9ED0(
                        (unsigned int)&v40,
                        a1,
                        a2,
                        *(_QWORD *)(v20 - 32LL * (*(_DWORD *)(v20 + 4) & 0x7FFFFFF)),
                        1,
                        0,
                        0);
                      sub_22EACA0(v39, a3, &v40);
                      sub_22C0090(a3);
                      sub_22C0650((__int64)a3, v39);
                      sub_22C0090(v39);
                      v18 = v36;
                      v19 = v34;
                      if ( v42 )
                        break;
                    }
                  }
                }
                v18 += 32;
                if ( v18 == v19 )
                  goto LABEL_14;
              }
              v42 = 0;
              sub_22C0090((unsigned __int8 *)&v40);
              v19 = v34;
              v18 = v36 + 32;
            }
            while ( v36 + 32 != v34 );
          }
        }
      }
      else
      {
        v29 = 1;
        while ( v17 != -4096 )
        {
          v30 = v29 + 1;
          v15 = v13 & (v29 + v15);
          v16 = v14 + 88LL * v15;
          v17 = *(_QWORD *)(v16 + 24);
          if ( a2 == v17 )
            goto LABEL_5;
          v29 = v30;
        }
        v40 = &unk_49DB368;
        sub_D68D70(v41);
      }
    }
  }
LABEL_14:
  v22 = *(_QWORD *)(a1 + 256);
  if ( v22 )
  {
    if ( *(_QWORD *)(v22 + 16) )
    {
      if ( *(_QWORD *)(v11 + 56) != v9 + 24 )
      {
        v23 = (_QWORD *)(v11 + 48);
        v24 = (_QWORD *)(*(_QWORD *)(v9 + 24) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (_QWORD *)(v11 + 48) != v24 )
        {
          do
          {
            while ( 1 )
            {
              if ( !v24 )
                BUG();
              if ( *((_BYTE *)v24 - 24) == 85 )
              {
                v25 = *(v24 - 7);
                if ( v25 )
                {
                  if ( !*(_BYTE *)v25 && *(_QWORD *)(v25 + 24) == v24[7] && *(_DWORD *)(v25 + 36) == 153 )
                  {
                    v26 = v24[-4 * (*((_DWORD *)v24 - 5) & 0x7FFFFFF) - 3];
                    if ( v26 )
                    {
                      v31 = v24;
                      v32 = v23;
                      sub_22C9ED0((unsigned int)&v40, a1, a2, v26, 1, 0, 0);
                      sub_22EACA0(v39, a3, &v40);
                      sub_22C0090(a3);
                      sub_22C0650((__int64)a3, v39);
                      sub_22C0090(v39);
                      v23 = v32;
                      v24 = v31;
                      if ( v42 )
                        break;
                    }
                  }
                }
              }
              v24 = (_QWORD *)(*v24 & 0xFFFFFFFFFFFFFFF8LL);
              if ( v23 == v24 )
                goto LABEL_29;
            }
            v42 = 0;
            sub_22C0090((unsigned __int8 *)&v40);
            v23 = v32;
            v24 = (_QWORD *)(*v31 & 0xFFFFFFFFFFFFFFF8LL);
          }
          while ( v32 != v24 );
        }
      }
    }
  }
LABEL_29:
  if ( *a3 == 6 && *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 14 )
  {
    v27 = *(_QWORD *)(v11 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v27 != v11 + 48 )
    {
      if ( !v27 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v27 - 24) - 30 <= 0xA && v9 == v27 - 24 )
      {
        v37 = *(__int64 ***)(a2 + 8);
        if ( (unsigned __int8)sub_22C65B0(a1, a2, v11) )
        {
          v28 = (unsigned __int8 *)sub_AC9EC0(v37);
          LOWORD(v40) = 0;
          sub_22C0430((__int64)&v40, v28);
          sub_22C0090(a3);
          sub_22C0650((__int64)a3, (unsigned __int8 *)&v40);
          sub_22C0090((unsigned __int8 *)&v40);
        }
      }
    }
  }
}

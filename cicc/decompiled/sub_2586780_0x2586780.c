// Function: sub_2586780
// Address: 0x2586780
//
void __fastcall sub_2586780(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r13
  __int64 *v4; // r12
  __int64 *v5; // rbx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int8 *v8; // rax
  unsigned __int8 *v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r12
  __int64 i; // rbx
  __int64 v14; // rcx
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 *v17; // rbx
  unsigned __int64 j; // r12
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // rcx
  unsigned __int64 v22; // rcx
  __int64 *v23; // rdi
  _BYTE *v24; // r15
  unsigned int v25; // eax
  __int64 *v26; // r9
  __int64 v27; // r10
  int v28; // eax
  _BYTE *v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  int v33; // r9d
  int v34; // r11d
  __int64 v35; // rdx
  _BYTE *v36; // [rsp+18h] [rbp-128h]
  __int64 v37; // [rsp+20h] [rbp-120h]
  __int64 v38; // [rsp+30h] [rbp-110h]
  __int64 *v39; // [rsp+48h] [rbp-F8h]
  _BYTE **v40; // [rsp+58h] [rbp-E8h] BYREF
  void *v41; // [rsp+60h] [rbp-E0h] BYREF
  unsigned __int64 v42; // [rsp+68h] [rbp-D8h]
  __int64 v43; // [rsp+70h] [rbp-D0h]
  __int64 *v44; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+88h] [rbp-B8h]
  _BYTE v46[32]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v48; // [rsp+B8h] [rbp-88h]
  __int64 v49; // [rsp+C0h] [rbp-80h]
  __int64 v50; // [rsp+C8h] [rbp-78h]
  __int64 *v51; // [rsp+D0h] [rbp-70h]
  __int64 v52; // [rsp+D8h] [rbp-68h]
  _BYTE *v53; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v54; // [rsp+E8h] [rbp-58h]
  _BYTE v55[80]; // [rsp+F0h] [rbp-50h] BYREF

  v3 = (_QWORD *)(a1 + 72);
  v44 = (__int64 *)v46;
  v45 = 0x400000000LL;
  LODWORD(v53) = 86;
  sub_2515D00(a2, (__m128i *)(a1 + 72), (int *)&v53, 1, (__int64)&v44, 0);
  v4 = v44;
  v5 = &v44[(unsigned int)v45];
  if ( v5 != v44 )
  {
    do
    {
      v6 = sub_A71B80(v4);
      v7 = v6;
      if ( *(_QWORD *)(a1 + 104) >= v6 )
        v7 = *(_QWORD *)(a1 + 104);
      if ( *(_QWORD *)(a1 + 96) >= v6 )
        v6 = *(_QWORD *)(a1 + 96);
      ++v4;
      *(_QWORD *)(a1 + 104) = v7;
      *(_QWORD *)(a1 + 96) = v6;
    }
    while ( v5 != v4 );
  }
  v8 = (unsigned __int8 *)sub_250D070(v3);
  v9 = sub_BD3990(v8, (__int64)v3);
  v10 = 1LL << sub_BD5420(v9, *(_QWORD *)(*(_QWORD *)(a2 + 208) + 104LL));
  v11 = v10;
  if ( *(_QWORD *)(a1 + 104) >= v10 )
    v10 = *(_QWORD *)(a1 + 104);
  if ( *(_QWORD *)(a1 + 96) >= v11 )
    v11 = *(_QWORD *)(a1 + 96);
  *(_QWORD *)(a1 + 104) = v10;
  *(_QWORD *)(a1 + 96) = v11;
  v12 = sub_2509740(v3);
  if ( v12 )
  {
    v38 = *(_QWORD *)(*(_QWORD *)(a2 + 208) + 120LL);
    if ( v38 )
    {
      v47 = 0;
      v48 = 0;
      v49 = 0;
      v50 = 0;
      v51 = (__int64 *)&v53;
      v52 = 0;
      for ( i = *(_QWORD *)(sub_250D070(v3) + 16); i; i = *(_QWORD *)(i + 8) )
      {
        v53 = (_BYTE *)i;
        sub_25789E0((__int64)&v47, (__int64 *)&v53);
      }
      sub_2586040(a1, a2, v38, v12, (__int64)&v47, a1 + 88);
      if ( *(_QWORD *)(a1 + 104) != *(_QWORD *)(a1 + 96) )
      {
        v53 = v55;
        v54 = 0x400000000LL;
        v40 = &v53;
        sub_2568920(v38, v12, (unsigned __int8 (__fastcall *)(__int64))sub_2535620, (__int64)&v40);
        v36 = &v53[8 * (unsigned int)v54];
        if ( v53 != v36 )
        {
          v37 = (__int64)v53;
          do
          {
            v14 = *(_QWORD *)v37;
            v15 = *(_DWORD *)(*(_QWORD *)v37 + 4LL) & 0x7FFFFFF;
            v16 = 32LL * v15;
            if ( (*(_BYTE *)(*(_QWORD *)v37 + 7LL) & 0x40) != 0 )
            {
              v17 = *(__int64 **)(v14 - 8);
              v39 = &v17[(unsigned __int64)v16 / 8];
              if ( v15 == 3 )
                v17 += 4;
            }
            else
            {
              v39 = *(__int64 **)v37;
              v17 = (__int64 *)(v14 - v16);
              v35 = v14 - v16 + 32;
              if ( v15 == 3 )
                v17 = (__int64 *)v35;
            }
            for ( j = 0x100000000LL; v39 != v17; v17 += 4 )
            {
              v19 = *v17;
              v20 = (unsigned int)v52;
              v42 = 1;
              v21 = *(_QWORD *)(v19 + 56);
              v43 = 0x100000000LL;
              if ( v21 )
                v21 -= 24;
              v41 = &unk_4A16ED8;
              sub_2586040(a1, a2, v38, v21, (__int64)&v47, (__int64)&v41);
              v22 = (unsigned __int64)v51;
              v23 = &v51[v20];
              v24 = v23 + 1;
              if ( v23 != &v51[(unsigned int)v52] )
              {
                do
                {
                  if ( (_DWORD)v50 )
                  {
                    v25 = (v50 - 1) & (((unsigned int)*v23 >> 9) ^ ((unsigned int)*v23 >> 4));
                    v26 = (__int64 *)(v48 + 8LL * v25);
                    v27 = *v26;
                    if ( *v23 == *v26 )
                    {
LABEL_27:
                      *v26 = -8192;
                      v22 = (unsigned __int64)v51;
                      LODWORD(v49) = v49 - 1;
                      ++HIDWORD(v49);
                    }
                    else
                    {
                      v33 = 1;
                      while ( v27 != -4096 )
                      {
                        v34 = v33 + 1;
                        v25 = (v50 - 1) & (v33 + v25);
                        v26 = (__int64 *)(v48 + 8LL * v25);
                        v27 = *v26;
                        if ( *v23 == *v26 )
                          goto LABEL_27;
                        v33 = v34;
                      }
                    }
                  }
                  v28 = v52;
                  v29 = (_BYTE *)(v22 + 8LL * (unsigned int)v52);
                  if ( v24 != v29 )
                  {
                    v30 = (__int64 *)memmove(v23, v24, v29 - v24);
                    v22 = (unsigned __int64)v51;
                    v23 = v30;
                    v28 = v52;
                  }
                  v31 = (unsigned int)(v28 - 1);
                  LODWORD(v52) = v31;
                }
                while ( v23 != (__int64 *)(v22 + 8 * v31) );
              }
              if ( j > v42 )
                j = v42;
            }
            v32 = j;
            if ( *(_QWORD *)(a1 + 104) >= j )
              v32 = *(_QWORD *)(a1 + 104);
            if ( *(_QWORD *)(a1 + 96) >= j )
              j = *(_QWORD *)(a1 + 96);
            *(_QWORD *)(a1 + 104) = v32;
            v37 += 8;
            *(_QWORD *)(a1 + 96) = j;
          }
          while ( v36 != (_BYTE *)v37 );
          v36 = v53;
        }
        if ( v36 != v55 )
          _libc_free((unsigned __int64)v36);
      }
      if ( v51 != (__int64 *)&v53 )
        _libc_free((unsigned __int64)v51);
      sub_C7D6A0(v48, 8LL * (unsigned int)v50, 8);
    }
  }
  if ( v44 != (__int64 *)v46 )
    _libc_free((unsigned __int64)v44);
}

// Function: sub_35407B0
// Address: 0x35407b0
//
void __fastcall sub_35407B0(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // rdx
  unsigned int v5; // ecx
  bool v6; // di
  __int64 v7; // rsi
  __int64 v8; // r11
  unsigned int v9; // r10d
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // r13
  char v14; // cl
  int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // rdx
  __int64 v19; // rbx
  char v20; // dl
  __int64 v21; // rdx
  unsigned __int64 v22; // rdi
  int v23; // edx
  int v24; // edx
  int v25; // eax
  char v26; // al
  _BYTE *v27; // rdi
  __int64 v28; // rdi
  char v29; // cl
  __int64 v30; // rbx
  __int64 v31; // rsi
  int v32; // edx
  __int64 v33; // rdi
  int v34; // eax
  int v35; // edx
  int v36; // edx
  char v37; // dl
  unsigned int v38; // ecx
  bool v39; // r8
  unsigned int v40; // ecx
  int v41; // edx
  int v42; // edx
  int v43; // edx
  int v44; // edx
  char v45; // dl
  _BYTE *v46; // rdi
  unsigned int v47; // ecx
  unsigned int v48; // esi
  int v49; // esi
  int v50; // ecx
  unsigned __int64 v51; // rdi
  _BYTE *v52; // rcx
  char v53; // dl
  unsigned __int64 v54; // rdi
  _BYTE *v55; // rdx
  char v56; // al
  __int64 v58; // [rsp+18h] [rbp-88h]
  __int64 v59; // [rsp+18h] [rbp-88h]
  unsigned __int64 v60; // [rsp+20h] [rbp-80h]
  unsigned __int64 v61; // [rsp+20h] [rbp-80h]
  int v62; // [rsp+28h] [rbp-78h]
  int v63; // [rsp+28h] [rbp-78h]
  _BYTE *v64; // [rsp+30h] [rbp-70h] BYREF
  __int64 v65; // [rsp+38h] [rbp-68h]
  _BYTE v66[4]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v67; // [rsp+44h] [rbp-5Ch]
  __int64 v68; // [rsp+4Ch] [rbp-54h]
  __int64 v69; // [rsp+58h] [rbp-48h]
  int v70; // [rsp+60h] [rbp-40h]

  if ( a1 != a2 && a2 != a1 + 88 )
  {
    v3 = a1 + 88;
    do
    {
      v4 = *(unsigned int *)(v3 + 52);
      v5 = *(_DWORD *)(a1 + 52);
      v6 = (unsigned int)v4 > v5;
      if ( (_DWORD)v4 == v5 )
      {
        v47 = *(_DWORD *)(v3 + 64);
        if ( !v47 || (v48 = *(_DWORD *)(a1 + 64), v47 == v48) || (v6 = v47 < v48, !v48) )
        {
          v49 = *(_DWORD *)(v3 + 56);
          v50 = *(_DWORD *)(a1 + 56);
          v6 = v49 < v50;
          if ( v49 == v50 )
            v6 = *(_DWORD *)(v3 + 60) > *(_DWORD *)(a1 + 60);
        }
      }
      v7 = *(unsigned int *)(v3 + 40);
      v8 = *(_QWORD *)(v3 + 8);
      v9 = *(_DWORD *)(v3 + 16);
      v10 = *(unsigned int *)(v3 + 20);
      v11 = *(unsigned int *)(v3 + 24);
      v12 = *(_QWORD *)v3 + 1LL;
      if ( v6 )
      {
        *(_QWORD *)v3 = v12;
        v58 = v8;
        *(_QWORD *)(v3 + 8) = 0;
        v60 = __PAIR64__(v10, v9);
        *(_DWORD *)(v3 + 16) = 0;
        *(_DWORD *)(v3 + 20) = 0;
        v62 = v11;
        *(_DWORD *)(v3 + 24) = 0;
        v64 = v66;
        v65 = 0;
        if ( (_DWORD)v7 )
        {
          sub_353DE10((__int64)&v64, (char **)(v3 + 32), v4, v12, v10, v11);
          LODWORD(v4) = *(_DWORD *)(v3 + 52);
        }
        LODWORD(v67) = v4;
        v13 = v3 + 88;
        v14 = *(_BYTE *)(v3 + 48);
        HIDWORD(v67) = *(_DWORD *)(v3 + 56);
        v15 = *(_DWORD *)(v3 + 60);
        v66[0] = v14;
        LODWORD(v68) = v15;
        HIDWORD(v68) = *(_DWORD *)(v3 + 64);
        v69 = *(_QWORD *)(v3 + 72);
        v70 = *(_DWORD *)(v3 + 80);
        v16 = v3;
        v17 = v3 - 56;
        v18 = v16 - a1;
        v19 = 0x2E8BA2E8BA2E8BA3LL * (v18 >> 3);
        if ( v18 > 0 )
        {
          do
          {
            sub_C7D6A0(*(_QWORD *)(v17 + 64), 8LL * *(unsigned int *)(v17 + 80), 8);
            v21 = *(_QWORD *)(v17 - 24);
            ++*(_QWORD *)(v17 + 56);
            ++*(_QWORD *)(v17 - 32);
            *(_QWORD *)(v17 + 64) = v21;
            LODWORD(v21) = *(_DWORD *)(v17 - 16);
            *(_QWORD *)(v17 - 24) = 0;
            *(_DWORD *)(v17 + 72) = v21;
            LODWORD(v21) = *(_DWORD *)(v17 - 12);
            *(_DWORD *)(v17 - 16) = 0;
            *(_DWORD *)(v17 + 76) = v21;
            LODWORD(v21) = *(_DWORD *)(v17 - 8);
            *(_DWORD *)(v17 - 12) = 0;
            *(_DWORD *)(v17 + 80) = v21;
            LODWORD(v21) = *(_DWORD *)(v17 + 8);
            *(_DWORD *)(v17 - 8) = 0;
            if ( (_DWORD)v21 )
            {
              v22 = *(_QWORD *)(v17 + 88);
              if ( v22 != v17 + 104 )
                _libc_free(v22);
              *(_QWORD *)(v17 + 88) = *(_QWORD *)v17;
              v23 = *(_DWORD *)(v17 + 8);
              *(_DWORD *)(v17 + 8) = 0;
              *(_DWORD *)(v17 + 96) = v23;
              v24 = *(_DWORD *)(v17 + 12);
              *(_DWORD *)(v17 + 12) = 0;
              *(_DWORD *)(v17 + 100) = v24;
              *(_QWORD *)v17 = v17 + 16;
            }
            else
            {
              *(_DWORD *)(v17 + 96) = 0;
            }
            v20 = *(_BYTE *)(v17 + 16);
            v17 -= 88;
            *(_BYTE *)(v17 + 192) = v20;
            *(_DWORD *)(v17 + 196) = *(_DWORD *)(v17 + 108);
            *(_DWORD *)(v17 + 200) = *(_DWORD *)(v17 + 112);
            *(_DWORD *)(v17 + 204) = *(_DWORD *)(v17 + 116);
            *(_DWORD *)(v17 + 208) = *(_DWORD *)(v17 + 120);
            *(_QWORD *)(v17 + 216) = *(_QWORD *)(v17 + 128);
            *(_DWORD *)(v17 + 224) = *(_DWORD *)(v17 + 136);
            --v19;
          }
          while ( v19 );
        }
        sub_C7D6A0(*(_QWORD *)(a1 + 8), 8LL * *(unsigned int *)(a1 + 24), 8);
        ++*(_QWORD *)a1;
        *(_QWORD *)(a1 + 8) = v58;
        *(_QWORD *)(a1 + 16) = v60;
        *(_DWORD *)(a1 + 24) = v62;
        v25 = v65;
        if ( (_DWORD)v65 )
        {
          v54 = *(_QWORD *)(a1 + 32);
          if ( v54 != a1 + 48 )
          {
            _libc_free(v54);
            v25 = v65;
          }
          *(_DWORD *)(a1 + 40) = v25;
          v55 = v64;
          *(_DWORD *)(a1 + 44) = HIDWORD(v65);
          v56 = v66[0];
          *(_QWORD *)(a1 + 32) = v55;
          *(_BYTE *)(a1 + 48) = v56;
          *(_QWORD *)(a1 + 52) = v67;
          *(_QWORD *)(a1 + 60) = v68;
          *(_QWORD *)(a1 + 72) = v69;
          *(_DWORD *)(a1 + 80) = v70;
        }
        else
        {
          v26 = v66[0];
          v27 = v64;
          *(_DWORD *)(a1 + 40) = 0;
          *(_BYTE *)(a1 + 48) = v26;
          *(_QWORD *)(a1 + 52) = v67;
          *(_QWORD *)(a1 + 60) = v68;
          *(_QWORD *)(a1 + 72) = v69;
          *(_DWORD *)(a1 + 80) = v70;
          if ( v27 != v66 )
            _libc_free((unsigned __int64)v27);
        }
        sub_C7D6A0(0, 0, 8);
      }
      else
      {
        v28 = 0;
        *(_QWORD *)v3 = v12;
        v59 = v8;
        *(_QWORD *)(v3 + 8) = 0;
        v61 = __PAIR64__(v10, v9);
        *(_DWORD *)(v3 + 16) = 0;
        *(_DWORD *)(v3 + 20) = 0;
        v63 = v11;
        *(_DWORD *)(v3 + 24) = 0;
        v64 = v66;
        v65 = 0;
        if ( (_DWORD)v7 )
        {
          sub_353DE10((__int64)&v64, (char **)(v3 + 32), v4, v12, v10, v11);
          LODWORD(v4) = *(_DWORD *)(v3 + 52);
          v7 = *(unsigned int *)(v3 + 24);
          v28 = *(_QWORD *)(v3 + 8);
        }
        v29 = *(_BYTE *)(v3 + 48);
        LODWORD(v67) = v4;
        v30 = v3;
        v66[0] = v29;
        HIDWORD(v67) = *(_DWORD *)(v3 + 56);
        v68 = *(_QWORD *)(v3 + 60);
        v69 = *(_QWORD *)(v3 + 72);
        v70 = *(_DWORD *)(v3 + 80);
        while ( 1 )
        {
          v38 = *(_DWORD *)(v30 - 36);
          v39 = v38 < (unsigned int)v4;
          if ( v38 == (_DWORD)v4 )
          {
            if ( !HIDWORD(v68) || (v40 = *(_DWORD *)(v30 - 24), HIDWORD(v68) == v40) || (v39 = HIDWORD(v68) < v40, !v40) )
            {
              v41 = *(_DWORD *)(v30 - 32);
              v39 = SHIDWORD(v67) < v41;
              if ( HIDWORD(v67) == v41 )
                v39 = (unsigned int)v68 > *(_DWORD *)(v30 - 28);
            }
          }
          v31 = 8 * v7;
          if ( !v39 )
            break;
          sub_C7D6A0(v28, v31, 8);
          v32 = *(_DWORD *)(v30 - 72);
          v33 = *(_QWORD *)(v30 - 80);
          *(_DWORD *)(v30 - 72) = 0;
          v34 = *(_DWORD *)(v30 - 48);
          ++*(_QWORD *)v30;
          *(_DWORD *)(v30 + 16) = v32;
          v35 = *(_DWORD *)(v30 - 68);
          *(_QWORD *)(v30 + 8) = v33;
          v28 = 0;
          *(_DWORD *)(v30 + 20) = v35;
          v36 = *(_DWORD *)(v30 - 64);
          ++*(_QWORD *)(v30 - 88);
          *(_QWORD *)(v30 - 80) = 0;
          *(_DWORD *)(v30 - 68) = 0;
          *(_DWORD *)(v30 + 24) = v36;
          *(_DWORD *)(v30 - 64) = 0;
          if ( v34 )
          {
            if ( *(_QWORD *)(v30 + 32) != v30 + 48 )
            {
              _libc_free(*(_QWORD *)(v30 + 32));
              v28 = *(_QWORD *)(v30 - 80);
            }
            *(_QWORD *)(v30 + 32) = *(_QWORD *)(v30 - 56);
            v42 = *(_DWORD *)(v30 - 48);
            *(_DWORD *)(v30 - 48) = 0;
            *(_DWORD *)(v30 + 40) = v42;
            v43 = *(_DWORD *)(v30 - 44);
            *(_DWORD *)(v30 - 44) = 0;
            *(_DWORD *)(v30 + 44) = v43;
            *(_QWORD *)(v30 - 56) = v30 - 40;
          }
          else
          {
            *(_DWORD *)(v30 + 40) = 0;
          }
          v37 = *(_BYTE *)(v30 - 40);
          v7 = *(unsigned int *)(v30 - 64);
          v30 -= 88;
          *(_BYTE *)(v30 + 136) = v37;
          *(_DWORD *)(v30 + 140) = *(_DWORD *)(v30 + 52);
          *(_DWORD *)(v30 + 144) = *(_DWORD *)(v30 + 56);
          *(_DWORD *)(v30 + 148) = *(_DWORD *)(v30 + 60);
          *(_DWORD *)(v30 + 152) = *(_DWORD *)(v30 + 64);
          *(_QWORD *)(v30 + 160) = *(_QWORD *)(v30 + 72);
          *(_DWORD *)(v30 + 168) = *(_DWORD *)(v30 + 80);
          LODWORD(v4) = v67;
        }
        sub_C7D6A0(v28, v31, 8);
        ++*(_QWORD *)v30;
        *(_QWORD *)(v30 + 8) = v59;
        *(_QWORD *)(v30 + 16) = v61;
        *(_DWORD *)(v30 + 24) = v63;
        v44 = v65;
        if ( (_DWORD)v65 )
        {
          v51 = *(_QWORD *)(v30 + 32);
          if ( v51 != v30 + 48 )
          {
            _libc_free(v51);
            v44 = v65;
          }
          *(_DWORD *)(v30 + 40) = v44;
          v52 = v64;
          *(_DWORD *)(v30 + 44) = HIDWORD(v65);
          v53 = v66[0];
          *(_QWORD *)(v30 + 32) = v52;
          *(_BYTE *)(v30 + 48) = v53;
          *(_QWORD *)(v30 + 52) = v67;
          *(_QWORD *)(v30 + 60) = v68;
          *(_QWORD *)(v30 + 72) = v69;
          *(_DWORD *)(v30 + 80) = v70;
        }
        else
        {
          v45 = v66[0];
          v46 = v64;
          *(_DWORD *)(v30 + 40) = 0;
          *(_BYTE *)(v30 + 48) = v45;
          *(_QWORD *)(v30 + 52) = v67;
          *(_QWORD *)(v30 + 60) = v68;
          *(_QWORD *)(v30 + 72) = v69;
          *(_DWORD *)(v30 + 80) = v70;
          if ( v46 != v66 )
            _libc_free((unsigned __int64)v46);
        }
        v13 = v3 + 88;
        sub_C7D6A0(0, 0, 8);
      }
      v3 = v13;
    }
    while ( a2 != v13 );
  }
}

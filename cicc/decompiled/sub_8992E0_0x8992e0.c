// Function: sub_8992E0
// Address: 0x8992e0
//
void __fastcall sub_8992E0(__int64 a1, __int64 *a2, int a3, _DWORD *a4)
{
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rbx
  char v9; // r12
  char v10; // al
  const __m128i *v11; // r15
  unsigned __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _QWORD *v17; // r12
  __int64 v18; // rbx
  char v19; // r13
  __m128i *v20; // rax
  __m128i *v21; // r15
  __int64 v22; // rax
  __m128i **v23; // r12
  __int64 *i; // rbx
  __m128i *v25; // r13
  __m128i *v26; // rax
  __int64 v27; // r15
  __int64 v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // r8
  __int64 v31; // rdi
  _QWORD *v32; // rax
  _QWORD *v33; // rax
  _QWORD *v34; // rax
  __int64 **v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 *v39; // r9
  __int64 v40; // [rsp+8h] [rbp-D8h]
  int v41; // [rsp+14h] [rbp-CCh]
  int v44; // [rsp+3Ch] [rbp-A4h] BYREF
  __int64 *v45; // [rsp+40h] [rbp-A0h] BYREF
  __m128i *v46; // [rsp+48h] [rbp-98h] BYREF
  __int64 v47[18]; // [rsp+50h] [rbp-90h] BYREF

  v6 = a2[3];
  if ( v6 )
    ((void (*)(void))sub_8992E0)();
  v7 = *a2;
  v40 = *a2;
  if ( !a3 )
  {
    if ( !v40 )
      return;
    v41 = 0;
    goto LABEL_8;
  }
  ++*(_DWORD *)(a1 + 168);
  if ( v7 )
  {
    v41 = *(_DWORD *)(sub_892BC0(v7) + 4);
LABEL_8:
    v8 = v40;
    while ( 1 )
    {
      v9 = *(_BYTE *)(*(_QWORD *)(v8 + 8) + 80LL);
      if ( a3 )
      {
        *(_DWORD *)(sub_892BC0(v8) + 4) = *(_DWORD *)(a1 + 168);
        if ( v9 != 19 )
          goto LABEL_11;
      }
      else if ( v9 != 19 )
      {
        goto LABEL_11;
      }
      LODWORD(v47[0]) = 0;
      v27 = *(_QWORD *)(*(_QWORD *)(v8 + 64) + 32LL);
      sub_88DA60((__int64 ***)v27, 1u);
      v28 = *(_QWORD *)(v27 + 24);
      if ( v28 )
      {
        v29 = *(_QWORD *)(v28 + 24);
        if ( v29 )
        {
          v30 = *(_QWORD *)(v29 + 24);
          if ( v30 )
          {
            v31 = *(_QWORD *)(v30 + 24);
            if ( v31 )
              sub_890340(v31);
            v32 = *(_QWORD **)v30;
            if ( *(_QWORD *)v30 )
            {
              do
              {
                *(_BYTE *)(v32[1] + 83LL) |= 0x40u;
                v32 = (_QWORD *)*v32;
              }
              while ( v32 );
            }
          }
          v33 = *(_QWORD **)v29;
          if ( *(_QWORD *)v29 )
          {
            do
            {
              *(_BYTE *)(v33[1] + 83LL) |= 0x40u;
              v33 = (_QWORD *)*v33;
            }
            while ( v33 );
          }
        }
        v34 = *(_QWORD **)v28;
        if ( *(_QWORD *)v28 )
        {
          do
          {
            *(_BYTE *)(v34[1] + 83LL) |= 0x40u;
            v34 = (_QWORD *)*v34;
          }
          while ( v34 );
        }
      }
      v35 = *(__int64 ***)v27;
      if ( *(_QWORD *)v27 )
      {
        do
        {
          *((_BYTE *)v35[1] + 83) |= 0x40u;
          v35 = (__int64 **)*v35;
        }
        while ( v35 );
      }
      sub_8992E0(0, v27, 0, v47);
      sub_863FC0(0, v27, v36, v37, v38, v39);
      v6 = LODWORD(v47[0]);
      if ( LODWORD(v47[0]) )
        *(_BYTE *)(v8 + 57) |= 2u;
      v10 = *(_BYTE *)(v8 + 57);
      if ( (v10 & 2) != 0 )
      {
        *(_BYTE *)(*(_QWORD *)(v8 + 64) + 266LL) |= 4u;
LABEL_11:
        v10 = *(_BYTE *)(v8 + 57);
        if ( (v10 & 2) != 0 )
        {
          *a4 = 1;
          v10 = *(_BYTE *)(v8 + 57);
        }
      }
      if ( (v10 & 1) != 0 )
      {
        *(_BYTE *)(v8 + 57) &= ~1u;
        switch ( v9 )
        {
          case 3:
            v11 = (const __m128i *)(v8 + 96);
            sub_7BC160(v8 + 96);
            v6 = 1;
            v12 = sub_65CFF0(0, 1);
            if ( (unsigned int)sub_8DC060(v12) )
              *(_WORD *)(v8 + 56) |= 0x202u;
            *(_QWORD *)(v8 + 80) = v12;
            if ( word_4F06418[0] != 9 )
            {
              v6 = (__int64)&dword_4F063F8;
              v12 = 706;
              sub_6851C0(0x2C2u, &dword_4F063F8);
              while ( word_4F06418[0] != 9 )
                sub_7B8B50(0x2C2u, &dword_4F063F8, v13, v14, v15, v16);
            }
            break;
          case 19:
            v11 = (const __m128i *)(v8 + 96);
            sub_7BC160(v8 + 96);
            v12 = v8;
            sub_88EF30(v8);
            if ( word_4F06418[0] != 9 )
            {
              v6 = (__int64)&dword_4F063F8;
              v12 = 706;
              sub_6851C0(0x2C2u, &dword_4F063F8);
              while ( word_4F06418[0] != 9 )
                sub_7B8B50(0x2C2u, &dword_4F063F8, v13, v14, v15, v16);
            }
            break;
          case 2:
            v11 = (const __m128i *)(v8 + 96);
            sub_7BC160(v8 + 96);
            v12 = v8;
            sub_88EE40(v8);
            if ( word_4F06418[0] != 9 )
            {
              v6 = (__int64)&dword_4F063F8;
              v12 = 706;
              sub_6851C0(0x2C2u, &dword_4F063F8);
              if ( word_4F06418[0] != 9 )
              {
                do
                  sub_7B8B50(0x2C2u, &dword_4F063F8, v13, v14, v15, v16);
                while ( word_4F06418[0] != 9 );
                sub_7B8B50(0x2C2u, &dword_4F063F8, v13, v14, v15, v16);
                if ( (*(_BYTE *)(v8 + 56) & 2) != 0 )
                  goto LABEL_20;
LABEL_62:
                sub_7AEA70(v11);
                v6 = 1;
                sub_879020((__int64)v11, 1);
                goto LABEL_20;
              }
            }
            break;
          default:
            sub_721090();
        }
        sub_7B8B50(v12, (unsigned int *)v6, v13, v14, v15, v16);
        if ( (*(_BYTE *)(v8 + 56) & 2) == 0 )
          goto LABEL_62;
      }
LABEL_20:
      *(_BYTE *)(*(_QWORD *)(v8 + 8) + 83LL) &= ~0x40u;
      v8 = *(_QWORD *)v8;
      if ( !v8 )
      {
        v17 = (_QWORD *)*a2;
        v18 = a2[4];
        v19 = (v41 != 0) & a3;
        if ( *a2 )
        {
LABEL_22:
          v20 = 0;
          do
          {
            while ( 1 )
            {
              v21 = v20;
              v20 = sub_8992B0((__int64)v17);
              if ( !v21 )
                break;
              v21[7].m128i_i64[0] = (__int64)v20;
              v17 = (_QWORD *)*v17;
              if ( !v17 )
                goto LABEL_26;
            }
            *(_QWORD *)(v18 + 8) = v20;
            v17 = (_QWORD *)*v17;
          }
          while ( v17 );
        }
LABEL_26:
        if ( v19 )
        {
          if ( *(_DWORD *)(a1 + 168) != v41 )
          {
            v22 = a2[4];
            if ( v22 )
            {
              v23 = *(__m128i ***)(v22 + 16);
              if ( v23 )
              {
                v45 = 0;
                v44 = 0;
                sub_892DC0(v40, &v45, &v46, 0, 0);
                for ( i = v45; i; i = (__int64 *)*i )
                  *(_DWORD *)(sub_892BC0((__int64)i) + 4) = v41;
                v25 = (__m128i *)sub_896D70(0, v40, 0);
                sub_892150(v47);
                v26 = (__m128i *)sub_743530(*v23, v25, (__int64)v45, 24580, &v44, v47);
                if ( !v44 )
                  *v23 = v26;
                sub_725130(v25->m128i_i64);
              }
            }
          }
        }
        return;
      }
    }
  }
  v17 = (_QWORD *)*a2;
  v18 = a2[4];
  if ( *a2 )
  {
    v41 = 0;
    v19 = 0;
    goto LABEL_22;
  }
}

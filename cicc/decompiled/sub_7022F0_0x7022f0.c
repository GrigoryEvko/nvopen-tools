// Function: sub_7022F0
// Address: 0x7022f0
//
__int64 __fastcall sub_7022F0(
        const __m128i *a1,
        _QWORD *a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        __int64 *a9,
        _DWORD *a10,
        _QWORD *a11,
        __int64 a12,
        int *a13,
        __int64 *a14)
{
  char v14; // r14
  char v15; // r12
  __int64 v17; // rax
  __int64 i; // rdx
  __int64 result; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 j; // r8
  __int64 v23; // r9
  int v24; // r11d
  __int64 v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdi
  int v30; // eax
  int v31; // eax
  int v32; // eax
  __int64 v33; // rax
  int v34; // eax
  int v35; // eax
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // [rsp+8h] [rbp-78h]
  __int64 v41; // [rsp+8h] [rbp-78h]
  __int64 v42; // [rsp+8h] [rbp-78h]
  __int64 v43; // [rsp+8h] [rbp-78h]
  __int64 v44; // [rsp+8h] [rbp-78h]
  int v45; // [rsp+10h] [rbp-70h]
  __int64 v46; // [rsp+10h] [rbp-70h]
  __int64 v47; // [rsp+10h] [rbp-70h]
  int v48; // [rsp+10h] [rbp-70h]
  __int64 v49; // [rsp+10h] [rbp-70h]
  __int64 v50; // [rsp+10h] [rbp-70h]
  int v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  int v53; // [rsp+18h] [rbp-68h]
  __int64 v54; // [rsp+18h] [rbp-68h]
  _QWORD *v55; // [rsp+20h] [rbp-60h]
  __int64 v56; // [rsp+28h] [rbp-58h]
  __int64 v57; // [rsp+28h] [rbp-58h]
  int v58; // [rsp+30h] [rbp-50h]
  int v59; // [rsp+30h] [rbp-50h]
  char v60; // [rsp+34h] [rbp-4Ch]
  __int64 v62[7]; // [rsp+48h] [rbp-38h] BYREF

  v14 = a6;
  v15 = a4;
  v60 = a5;
  if ( a14 )
    *a14 = 0;
  if ( a1[1].m128i_i8[0] )
  {
    v17 = a1->m128i_i64[0];
    for ( i = *(unsigned __int8 *)(a1->m128i_i64[0] + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v17 + 140) )
      v17 = *(_QWORD *)(v17 + 160);
    if ( (_BYTE)i )
    {
      v55 = (_QWORD *)sub_6F6F40(a1, 0, i, a4, a5, a6);
      v56 = *v55;
      v20 = *v55;
      if ( (unsigned int)sub_8D3D10(*v55) )
      {
        v57 = sub_8D4870(v20);
        if ( (a1[1].m128i_i8[2] & 1) != 0 )
          goto LABEL_11;
        v26 = a3;
        v24 = 0;
      }
      else
      {
        v24 = sub_8D2EF0(v20);
        if ( v24 )
        {
          v57 = sub_8D46C0(v56);
          if ( (a1[1].m128i_i8[2] & 1) != 0 )
          {
LABEL_11:
            v21 = v57;
            if ( *(_BYTE *)(v57 + 140) == 12 )
            {
              do
                v21 = *(_QWORD *)(v21 + 160);
              while ( *(_BYTE *)(v21 + 140) == 12 );
            }
            else
            {
              v21 = v57;
            }
            v52 = *(_QWORD *)(*(_QWORD *)(v21 + 168) + 40LL);
            if ( v52 )
              v52 = sub_8D71D0(v57);
            for ( j = sub_8D46C0(v52); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
              ;
            v23 = *a2;
            v24 = 0;
            if ( (*((_BYTE *)a2 + 18) & 2) != 0 )
            {
              v43 = j;
              v49 = *a2;
              v35 = sub_8D3320(*a2);
              v23 = v49;
              j = v43;
              v24 = 1;
              if ( v35 )
              {
                v36 = sub_8D46C0(v49);
                j = v43;
                v24 = 1;
                v23 = v36;
              }
            }
            if ( dword_4F04C44 != -1
              || (v25 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v25 + 6) & 6) != 0)
              || *(_BYTE *)(v25 + 4) == 12 )
            {
              v58 = v24;
              v40 = j;
              v46 = v23;
              v30 = sub_8DBE70(v23);
              v23 = v46;
              j = v40;
              v24 = v58;
              if ( v30 || (v44 = v46, v50 = j, v37 = sub_8DBE70(v52), j = v50, v23 = v44, v24 = v58, v37) )
              {
                if ( *(_BYTE *)(j + 140) != 9 || (*(_BYTE *)(*(_QWORD *)(j + 168) + 109LL) & 0x20) == 0 )
                  goto LABEL_24;
                if ( v23 != j )
                {
                  v59 = v24;
                  v41 = v23;
                  v47 = j;
                  v31 = sub_8D97D0(j, v23, 32, v27, j);
                  v24 = v59;
                  if ( !v31 )
                  {
LABEL_24:
                    if ( *((_BYTE *)a2 + 16) == 2 && *((_BYTE *)a2 + 317) == 10 )
                    {
                      v51 = v24;
                      v54 = sub_6ECAE0(*a2, 0, 0, 1, 2u, (_QWORD *)((char *)a2 + 68), v62);
                      v38 = sub_73A460(a2 + 18);
                      sub_72F900(v62[0], v38);
                      v24 = v51;
                      v26 = (_QWORD *)v54;
                    }
                    else
                    {
                      v53 = v24;
                      v28 = sub_6F6F40((const __m128i *)a2, 0, (__int64)v26, v27, j, v23);
                      v24 = v53;
                      v26 = (_QWORD *)v28;
                    }
                    v26[2] = a3;
                    goto LABEL_28;
                  }
                  j = v47;
                  v23 = v41;
                }
              }
              if ( dword_4F04C44 != -1 )
                goto LABEL_42;
              v25 = qword_4F04C68[0] + 776LL * dword_4F04C64;
              if ( (*(_BYTE *)(v25 + 6) & 6) != 0 )
                goto LABEL_42;
            }
            if ( *(_BYTE *)(v25 + 4) == 12 )
            {
LABEL_42:
              while ( *(_BYTE *)(v23 + 140) == 12 )
                v23 = *(_QWORD *)(v23 + 160);
              if ( v23 != j )
              {
                v48 = v24;
                v42 = j;
                v32 = sub_8DED30(v23, j, 1);
                v24 = v48;
                if ( !v32 )
                {
                  if ( dword_4F04C44 == -1 )
                  {
                    v26 = qword_4F04C68;
                    v33 = qword_4F04C68[0] + 776LL * dword_4F04C64;
                    if ( (*(_BYTE *)(v33 + 6) & 6) == 0 && *(_BYTE *)(v33 + 4) != 12 )
                    {
                      v34 = sub_8DBE70(v42);
                      v24 = v48;
                      if ( !v34 )
                        sub_721090(v42);
                    }
                  }
                  goto LABEL_24;
                }
              }
            }
            v45 = v24;
            if ( v24 )
            {
              sub_6FC3F0(v52, (__m128i *)a2, 1u);
              v24 = v45;
            }
            else
            {
              v39 = sub_8D46C0(v52);
              sub_831640(a2, v39, 0);
              v24 = 0;
            }
            goto LABEL_24;
          }
          v26 = a3;
          v24 = 0;
        }
        else
        {
          v26 = a3;
          v57 = dword_4D03B80;
          if ( (a1[1].m128i_i8[2] & 1) != 0 )
          {
            if ( (*((_BYTE *)a2 + 18) & 2) != 0 )
            {
              v29 = *a2;
              if ( (unsigned int)sub_8D3320(*a2) )
                sub_8D46C0(v29);
              v24 = 1;
            }
            goto LABEL_24;
          }
        }
      }
LABEL_28:
      v55[2] = v26;
      sub_701D00(
        v55,
        v57,
        (a1[1].m128i_i8[2] & 4) != 0,
        (a1[1].m128i_i8[2] & 0x40) != 0,
        v24,
        v15,
        0,
        v60,
        v14,
        a7,
        a8,
        a9,
        a10,
        a11,
        (__m128i *)a12,
        a13,
        a14);
      goto LABEL_8;
    }
  }
  sub_6E6260((_QWORD *)a12);
LABEL_8:
  result = *(_QWORD *)a10;
  *(_QWORD *)(a12 + 68) = *(_QWORD *)a10;
  return result;
}

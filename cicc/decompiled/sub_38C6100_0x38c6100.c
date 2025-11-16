// Function: sub_38C6100
// Address: 0x38c6100
//
__int64 __fastcall sub_38C6100(__int64 *a1, __int64 *a2, int a3, __int64 a4)
{
  __int64 *v4; // r13
  __int64 *v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rbx
  int v8; // r13d
  int v9; // ecx
  __int64 v10; // rax
  __int64 *v11; // r15
  char v12; // r13
  char v13; // al
  __int64 v14; // r13
  unsigned int v15; // r14d
  __int64 v16; // r12
  int v17; // eax
  unsigned __int16 v18; // r14
  unsigned int v19; // eax
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // r12
  __int64 v25; // [rsp+8h] [rbp-78h]
  __int64 *v26; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+18h] [rbp-68h]
  __int64 *v28; // [rsp+20h] [rbp-60h]
  __int64 v29; // [rsp+28h] [rbp-58h]
  char v30; // [rsp+30h] [rbp-50h]
  unsigned __int8 v31; // [rsp+34h] [rbp-4Ch]
  int v32; // [rsp+34h] [rbp-4Ch]
  __int64 v33; // [rsp+38h] [rbp-48h]
  int v34; // [rsp+40h] [rbp-40h]
  int v35; // [rsp+40h] [rbp-40h]
  int v36; // [rsp+44h] [rbp-3Ch]

  v4 = a2;
  v5 = a2;
  sub_38C6010(a1, a2, (unsigned __int16)a3 | (BYTE2(a3) << 16), a4);
  v25 = v6;
  v26 = (__int64 *)a1[67];
  v28 = (__int64 *)a1[66];
  if ( v28 != v26 )
  {
    do
    {
      v27 = *v28;
      v29 = v28[2];
      if ( v28[1] == v29 )
      {
        v14 = 0;
      }
      else
      {
        v33 = 0;
        v7 = v28[1];
        v8 = 0;
        v9 = 0;
        v30 = 1;
        v10 = 1;
        v11 = v5;
        v36 = 1;
        while ( 1 )
        {
          v15 = *(_DWORD *)v7;
          v16 = *(unsigned int *)(v7 + 4) - v10;
          v17 = v36;
          v36 = *(_DWORD *)v7;
          if ( *(_DWORD *)v7 != v17 )
          {
            v34 = v9;
            (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v11 + 424))(v11, 4, 1);
            sub_38DCDD0(v11, v15);
            v9 = v34;
          }
          v35 = *(unsigned __int16 *)(v7 + 8);
          v18 = *(_WORD *)(v7 + 8);
          if ( v35 != v9 )
          {
            (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v11 + 424))(v11, 5, 1);
            sub_38DCDD0(v11, v18);
          }
          v19 = *(_DWORD *)(v7 + 12);
          if ( v19 && *(_WORD *)(v11[1] + 1160) > 3u )
          {
            v20 = v19;
            v32 = sub_3946290(v19);
            (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v11 + 424))(v11, 0, 1);
            sub_38DCDD0(v11, (unsigned int)(v32 + 1));
            (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v11 + 424))(v11, 4, 1);
            sub_38DCDD0(v11, v20);
          }
          v31 = *(_BYTE *)(v7 + 11);
          if ( v31 != v8 )
          {
            (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v11 + 424))(v11, 12, 1);
            sub_38DCDD0(v11, v31);
          }
          v12 = *(_BYTE *)(v7 + 10);
          v13 = v12;
          if ( (((unsigned __int8)v12 ^ (unsigned __int8)v30) & 1) != 0 )
          {
            (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v11 + 424))(v11, 6, 1);
            v30 = v12;
            v13 = *(_BYTE *)(v7 + 10);
          }
          if ( (v13 & 2) != 0 )
          {
            (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v11 + 424))(v11, 7, 1);
            v13 = *(_BYTE *)(v7 + 10);
          }
          if ( (v13 & 4) != 0 )
          {
            (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v11 + 424))(v11, 10, 1);
            v13 = *(_BYTE *)(v7 + 10);
          }
          if ( (v13 & 8) != 0 )
            (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v11 + 424))(v11, 11, 1);
          v14 = *(_QWORD *)(v7 + 16);
          v7 += 24;
          sub_38D59D0(v11, v16, v33, v14, *(unsigned int *)(*(_QWORD *)(v11[1] + 16) + 8LL));
          v10 = *(unsigned int *)(v7 - 20);
          if ( v29 == v7 )
            break;
          v9 = v35;
          v33 = v14;
          v8 = v31;
        }
        v5 = v11;
      }
      v21 = sub_38DDE50(v5, v27);
      v22 = v5[1];
      v23 = v21;
      (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*v5 + 160))(v5, *(_QWORD *)(*(_QWORD *)(v22 + 32) + 88LL), 0);
      sub_38D59D0(v5, 0x7FFFFFFFFFFFFFFFLL, v14, v23, *(unsigned int *)(*(_QWORD *)(v22 + 16) + 8LL));
      v28 += 4;
    }
    while ( v26 != v28 );
    v4 = v5;
  }
  return (*(__int64 (__fastcall **)(__int64 *, __int64, _QWORD))(*v4 + 176))(v4, v25, 0);
}

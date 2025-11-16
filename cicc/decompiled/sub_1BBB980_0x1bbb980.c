// Function: sub_1BBB980
// Address: 0x1bbb980
//
void __fastcall sub_1BBB980(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  char *v7; // r14
  char *v8; // r13
  __int64 v9; // rbx
  char *v11; // rax
  char *v12; // r10
  char *v13; // r11
  char *v14; // rax
  __int64 v15; // r13
  char *v16; // rax
  char *v17; // r11
  char *v18; // rax
  unsigned int v19; // edx
  __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // rdi
  char *v26; // [rsp+10h] [rbp-50h]
  char *src; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+20h] [rbp-40h]
  char *v29; // [rsp+20h] [rbp-40h]
  char *v30; // [rsp+20h] [rbp-40h]
  __int64 v31; // [rsp+28h] [rbp-38h]
  char *v32; // [rsp+28h] [rbp-38h]

  if ( !a4 )
    return;
  v6 = a5;
  if ( !a5 )
    return;
  v7 = a1;
  v8 = a2;
  v9 = a4;
  if ( a4 + a5 == 2 )
  {
    v17 = a2;
    v16 = a1;
LABEL_20:
    v23 = *(_QWORD *)v17;
    v21 = *(_QWORD *)v16;
    if ( *(_QWORD *)v16 != 0 && *(_QWORD *)v17 != 0 && v21 != v23 )
    {
      v22 = *(_QWORD *)(v21 + 8);
      if ( v23 != v22 )
      {
        if ( v21 == *(_QWORD *)(v23 + 8) || *(_DWORD *)(v23 + 16) >= *(_DWORD *)(v21 + 16) )
          return;
        v24 = *(_QWORD *)(a6 + 1352);
        if ( *(_BYTE *)(v24 + 72) )
        {
          if ( *(_DWORD *)(v21 + 48) < *(_DWORD *)(v23 + 48) || *(_DWORD *)(v21 + 52) > *(_DWORD *)(v23 + 52) )
            return;
          v22 = *(_QWORD *)v17;
        }
        else
        {
          v19 = *(_DWORD *)(v24 + 76) + 1;
          *(_DWORD *)(v24 + 76) = v19;
          if ( v19 > 0x20 )
          {
            v30 = v16;
            v32 = v17;
            sub_15CC640(v24);
            if ( *(_DWORD *)(v21 + 48) < *(_DWORD *)(v23 + 48) || *(_DWORD *)(v21 + 52) > *(_DWORD *)(v23 + 52) )
              return;
            v16 = v30;
            v17 = v32;
            v21 = *(_QWORD *)v30;
            v22 = *(_QWORD *)v32;
          }
          else
          {
            do
            {
              v20 = v21;
              v21 = *(_QWORD *)(v21 + 8);
            }
            while ( v21 && *(_DWORD *)(v23 + 16) <= *(_DWORD *)(v21 + 16) );
            if ( v23 != v20 )
              return;
            v21 = *(_QWORD *)v16;
            v22 = *(_QWORD *)v17;
          }
        }
      }
      *(_QWORD *)v16 = v22;
      *(_QWORD *)v17 = v21;
    }
  }
  else
  {
    if ( a5 >= a4 )
      goto LABEL_10;
LABEL_5:
    v28 = v9 / 2;
    v11 = (char *)sub_1BBA9A0(v8, a3, (__int64 *)&v7[8 * (v9 / 2)], a6);
    v12 = &v7[8 * (v9 / 2)];
    v13 = v11;
    v31 = (v11 - v8) >> 3;
    while ( 1 )
    {
      v26 = v13;
      src = v12;
      v14 = sub_1BBB7C0(v12, v8, v13);
      v15 = v28;
      v29 = v14;
      sub_1BBB980(v7, src, v14, v15, v31, a6);
      v6 -= v31;
      v9 -= v15;
      if ( !v9 )
        break;
      v16 = v29;
      v17 = v26;
      if ( !v6 )
        break;
      if ( v6 + v9 == 2 )
        goto LABEL_20;
      v8 = v26;
      v7 = v29;
      if ( v6 < v9 )
        goto LABEL_5;
LABEL_10:
      v31 = v6 / 2;
      v18 = (char *)sub_1BBAAE0(v7, (__int64)v8, (__int64 *)&v8[8 * (v6 / 2)], a6);
      v13 = &v8[8 * (v6 / 2)];
      v12 = v18;
      v28 = (v18 - v7) >> 3;
    }
  }
}

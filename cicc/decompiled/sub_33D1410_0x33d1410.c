// Function: sub_33D1410
// Address: 0x33d1410
//
char __fastcall sub_33D1410(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // eax
  unsigned __int16 *v7; // rdx
  int v8; // eax
  __int64 v9; // rdx
  unsigned __int16 v10; // ax
  char *v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // ebx
  char result; // al
  __int64 *v15; // rax
  __int64 v16; // rcx
  int v17; // edx
  __int64 v18; // rsi
  __int64 v19; // rdx
  unsigned __int16 *v20; // rdx
  int v21; // eax
  __int64 v22; // rdx
  unsigned __int16 v23; // ax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned int v27; // eax
  bool v28; // zf
  char v29; // [rsp+Fh] [rbp-91h]
  unsigned __int64 v30; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v31; // [rsp+18h] [rbp-88h]
  __int16 v32; // [rsp+20h] [rbp-80h] BYREF
  __int64 v33; // [rsp+28h] [rbp-78h]
  unsigned __int16 v34; // [rsp+30h] [rbp-70h] BYREF
  __int64 v35; // [rsp+38h] [rbp-68h]
  char *v36; // [rsp+40h] [rbp-60h] BYREF
  __int64 v37; // [rsp+48h] [rbp-58h]
  char *v38; // [rsp+50h] [rbp-50h] BYREF
  __int64 v39; // [rsp+58h] [rbp-48h]
  char v40; // [rsp+60h] [rbp-40h]

  v6 = *(_DWORD *)(a1 + 24);
  if ( v6 != 168 )
  {
    if ( v6 != 156 )
      return 0;
    v31 = 1;
    v7 = *(unsigned __int16 **)(a1 + 48);
    v30 = 0;
    v8 = *v7;
    v9 = *((_QWORD *)v7 + 1);
    v32 = v8;
    v33 = v9;
    if ( (_WORD)v8 )
    {
      v10 = word_4456580[v8 - 1];
      v35 = 0;
      v34 = v10;
      if ( !v10 )
      {
LABEL_5:
        v11 = (char *)sub_3007260((__int64)&v34);
        v38 = v11;
        v39 = v12;
LABEL_6:
        LOBYTE(v37) = v12;
        v36 = v11;
        v13 = sub_CA1930(&v36);
        result = sub_33D0890(a1, a2, &v30, (unsigned int *)&v36, (bool *)&v34, v13, 0);
        if ( result )
          result = (_DWORD)v36 == v13;
        if ( v31 > 0x40 )
        {
          if ( v30 )
          {
            v29 = result;
            j_j___libc_free_0_0(v30);
            return v29;
          }
        }
        return result;
      }
    }
    else
    {
      v10 = sub_3009970((__int64)&v32, a2, v9, a4, a5);
      v34 = v10;
      v35 = v19;
      if ( !v10 )
        goto LABEL_5;
    }
    if ( v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
      BUG();
    v12 = 16LL * (v10 - 1);
    v11 = *(char **)&byte_444C4A0[v12];
    LOBYTE(v12) = byte_444C4A0[v12 + 8];
    goto LABEL_6;
  }
  v15 = *(__int64 **)(a1 + 40);
  v16 = *v15;
  v17 = *(_DWORD *)(*v15 + 24);
  if ( v17 == 11 || v17 == 35 )
  {
    v18 = *(_QWORD *)(v16 + 96);
    LODWORD(v39) = *(_DWORD *)(v18 + 32);
    if ( (unsigned int)v39 > 0x40 )
    {
      v18 += 24;
      sub_C43780((__int64)&v38, (const void **)v18);
    }
    else
    {
      v38 = *(char **)(v18 + 24);
    }
    v40 = 1;
  }
  else
  {
    result = v17 == 12 || v17 == 36;
    if ( !result )
      return result;
    v18 = *(_QWORD *)(v16 + 96) + 24LL;
    if ( *(void **)v18 == sub_C33340() )
      sub_C3E660((__int64)&v36, v18);
    else
      sub_C3A850((__int64)&v36, (__int64 *)v18);
    v40 = 1;
    LODWORD(v39) = v37;
    v38 = v36;
  }
  v20 = *(unsigned __int16 **)(a1 + 48);
  v21 = *v20;
  v22 = *((_QWORD *)v20 + 1);
  v32 = v21;
  v33 = v22;
  if ( (_WORD)v21 )
  {
    v23 = word_4456580[v21 - 1];
    v24 = 0;
  }
  else
  {
    v23 = sub_3009970((__int64)&v32, v18, v22, v16, a5);
  }
  v35 = v24;
  v34 = v23;
  v25 = sub_2D5B750(&v34);
  v37 = v26;
  v36 = (char *)v25;
  v27 = sub_CA1930(&v36);
  sub_C44740((__int64)&v36, &v38, v27);
  if ( *(_DWORD *)(a2 + 8) > 0x40u && *(_QWORD *)a2 )
    j_j___libc_free_0_0(*(_QWORD *)a2);
  v28 = v40 == 0;
  *(_QWORD *)a2 = v36;
  *(_DWORD *)(a2 + 8) = v37;
  if ( !v28 )
  {
    v40 = 0;
    if ( (unsigned int)v39 > 0x40 )
    {
      if ( v38 )
        j_j___libc_free_0_0((unsigned __int64)v38);
    }
  }
  return 1;
}

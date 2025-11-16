// Function: sub_33CACD0
// Address: 0x33cacd0
//
__int64 __fastcall sub_33CACD0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, char a6, char a7)
{
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // rax
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned __int16 v17; // r10
  __int64 v18; // rax
  __int64 v19; // r15
  unsigned int *v20; // r8
  __int64 v21; // rcx
  int v22; // edx
  int v23; // eax
  bool v24; // r10
  bool v25; // r11
  bool v26; // al
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // rax
  __int16 v30; // cx
  __int64 v31; // rax
  __int64 v32; // r8
  bool v33; // zf
  bool v34; // al
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  unsigned __int16 v38; // ax
  __int64 v39; // rdx
  __int64 v40; // [rsp+0h] [rbp-80h]
  __int64 v41; // [rsp+8h] [rbp-78h]
  unsigned __int16 v42; // [rsp+14h] [rbp-6Ch]
  __int64 v43; // [rsp+28h] [rbp-58h] BYREF
  __int64 v44; // [rsp+30h] [rbp-50h] BYREF
  __int64 v45; // [rsp+38h] [rbp-48h] BYREF
  _QWORD v46[8]; // [rsp+40h] [rbp-40h] BYREF

  v10 = a1;
  if ( !a7 )
  {
    v11 = *(_QWORD *)(a3 + 48) + 16LL * a4;
    v12 = *(_QWORD *)(a1 + 48) + 16LL * (unsigned int)a2;
    if ( *(_WORD *)v12 != *(_WORD *)v11 )
      return 0;
    a1 = *(_QWORD *)(v11 + 8);
    if ( *(_QWORD *)(v12 + 8) != a1 && !*(_WORD *)v12 )
      return 0;
  }
  v14 = *(_DWORD *)(v10 + 24);
  v15 = *(unsigned int *)(a3 + 24);
  if ( v14 != 35 && v14 != 11 || (_DWORD)v15 != 11 && (_DWORD)v15 != 35 )
  {
    if ( v14 != (_DWORD)v15 || v14 != 156 && v14 != 168 )
      return 0;
    v16 = *(_QWORD *)(v10 + 48) + 16LL * (unsigned int)a2;
    v17 = *(_WORD *)v16;
    v41 = *(_QWORD *)(v16 + 8);
    LOWORD(v46[0]) = v17;
    v46[1] = v41;
    if ( v17 )
    {
      if ( (unsigned __int16)(v17 - 17) <= 0xD3u )
      {
        v41 = 0;
        v17 = word_4456580[v17 - 1];
      }
    }
    else
    {
      v34 = sub_30070B0((__int64)v46);
      v17 = 0;
      if ( v34 )
      {
        v38 = sub_3009970((__int64)v46, v16, v35, v36, v37);
        v41 = v39;
        v17 = v38;
      }
    }
    v18 = *(unsigned int *)(v10 + 64);
    if ( !(_DWORD)v18 )
      return 1;
    v42 = v17;
    v19 = 0;
    v40 = 40 * v18;
    while ( 1 )
    {
      a1 = v19 + *(_QWORD *)(v10 + 40);
      v20 = (unsigned int *)(v19 + *(_QWORD *)(a3 + 40));
      v21 = *(_QWORD *)a1;
      a2 = *(_QWORD *)v20;
      v22 = *(_DWORD *)(*(_QWORD *)a1 + 24LL);
      v23 = *(_DWORD *)(*(_QWORD *)v20 + 24LL);
      if ( a6 )
      {
        v24 = v22 == 51;
        v25 = v23 == 51;
      }
      else
      {
        v24 = 0;
        v25 = 0;
      }
      v26 = v23 == 11 || v23 == 35;
      if ( v22 == 35 || v22 == 11 )
      {
        v27 = *(_QWORD *)a1;
        v15 = *(_QWORD *)v20;
        if ( v26 )
          goto LABEL_22;
      }
      else
      {
        if ( v26 )
        {
          if ( !v24 )
            return 0;
          v15 = *(_QWORD *)v20;
          v27 = 0;
          goto LABEL_22;
        }
        if ( !v24 )
          return 0;
        v27 = 0;
      }
      if ( !v25 )
        return 0;
      v15 = 0;
LABEL_22:
      if ( !a7 )
      {
        v28 = *(unsigned int *)(a1 + 8);
        a1 = v42;
        v29 = *(_QWORD *)(v21 + 48) + 16 * v28;
        v30 = *(_WORD *)v29;
        if ( v42 != *(_WORD *)v29 )
          return 0;
        v31 = *(_QWORD *)(v29 + 8);
        LOBYTE(a1) = v42 == 0;
        if ( v41 != v31 && !v42 )
          return 0;
        v32 = *(_QWORD *)(a2 + 48) + 16LL * v20[2];
        if ( v30 != *(_WORD *)v32 || *(_QWORD *)(v32 + 8) != v31 && !v42 )
          return 0;
      }
      v33 = *(_QWORD *)(a5 + 16) == 0;
      v45 = v27;
      v46[0] = v15;
      if ( v33 )
        goto LABEL_47;
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64 *, _QWORD *))(a5 + 24))(a5, &v45, v46) )
        return 0;
      v19 += 40;
      if ( v40 == v19 )
        return 1;
    }
  }
  v33 = *(_QWORD *)(a5 + 16) == 0;
  v43 = v10;
  v44 = a3;
  if ( v33 )
LABEL_47:
    sub_4263D6(a1, a2, v15);
  return (*(__int64 (__fastcall **)(__int64, __int64 *, __int64 *))(a5 + 24))(a5, &v43, &v44);
}

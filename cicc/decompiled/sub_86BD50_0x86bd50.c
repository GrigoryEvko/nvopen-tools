// Function: sub_86BD50
// Address: 0x86bd50
//
unsigned __int8 *__fastcall sub_86BD50(__int64 *a1, __int64 a2, _DWORD *a3, __int64 *a4, unsigned __int8 *a5)
{
  __int64 *v5; // r15
  __int64 v6; // r14
  unsigned __int8 *result; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned __int8 v12; // r13
  unsigned __int8 v13; // al
  unsigned __int8 v14; // r11
  _DWORD *v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // r13
  int v18; // eax
  __int64 v19; // rax
  int v20; // eax
  unsigned __int8 v21; // [rsp+4h] [rbp-4Ch]
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+18h] [rbp-38h]
  __int64 v29; // [rsp+18h] [rbp-38h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  __int64 v31; // [rsp+18h] [rbp-38h]
  __int64 v32; // [rsp+18h] [rbp-38h]

  v5 = a1;
  v6 = a2;
  result = *(unsigned __int8 **)(a2 + 16);
  if ( result != (unsigned __int8 *)a1[2] )
  {
    a2 = *((_QWORD *)result + 1);
    sub_86BD50(a1, a2);
    result = *(unsigned __int8 **)(v6 + 16);
    v5 = *(__int64 **)result;
  }
  while ( 1 )
  {
    if ( *((_BYTE *)v5 + 32) != 1 )
      goto LABEL_3;
    v10 = v5[6];
    v11 = v5[5];
    if ( v10 && (v5[7] & 1) == 0 )
    {
      if ( *(_BYTE *)(v10 + 136) <= 2u )
        goto LABEL_6;
      if ( dword_4F077C4 != 2 )
        goto LABEL_13;
      v17 = *(_QWORD *)(v10 + 120);
      v25 = v5[5];
      v30 = v5[6];
      v18 = sub_8D3410(v17);
      v10 = v30;
      v11 = v25;
      if ( v18 )
      {
        v26 = v30;
        v31 = v11;
        v19 = sub_8D40F0(v17);
        v10 = v26;
        v11 = v31;
        v17 = v19;
      }
      while ( *(_BYTE *)(v17 + 140) == 12 )
        v17 = *(_QWORD *)(v17 + 160);
      if ( v11 && (HIDWORD(qword_4F077B4) || (v27 = v10, v32 = v11, v20 = sub_8D3A70(v17), v11 = v32, v10 = v27, v20)) )
      {
        v12 = 8;
      }
      else
      {
        if ( !dword_4D04964 )
        {
LABEL_13:
          v12 = 5;
          goto LABEL_14;
        }
        result = byte_4F07472;
        v12 = byte_4F07472[0];
        if ( byte_4F07472[0] == 3 )
          goto LABEL_3;
      }
LABEL_14:
      v13 = *a5;
      v14 = v12;
      if ( v12 == *a5 )
        goto LABEL_19;
      goto LABEL_15;
    }
    if ( *(_BYTE *)(v11 + 40) == 22 && !*(_BYTE *)(v11 + 72) )
      v10 = *(_QWORD *)(v11 + 80);
    v13 = *a5;
    if ( *a5 != 8 )
    {
      v14 = 8;
      v12 = 8;
LABEL_15:
      if ( v13 != 3 )
      {
        v21 = v14;
        v23 = v10;
        v28 = v11;
        sub_685910(*a4, (FILE *)a2);
        v14 = v21;
        v10 = v23;
        v11 = v28;
      }
      v24 = v10;
      v29 = v11;
      v15 = sub_67D910(v14, 0x222u, a3);
      v10 = v24;
      v11 = v29;
      *a4 = (__int64)v15;
      *a5 = v12;
    }
    if ( !v10 )
    {
      a2 = 895;
      result = (unsigned __int8 *)sub_67DDB0((_QWORD *)*a4, 895, (_QWORD *)v11);
LABEL_3:
      if ( (__int64 *)v6 == v5 )
        return result;
      goto LABEL_4;
    }
LABEL_19:
    v16 = (_QWORD *)*a4;
    if ( (*(_BYTE *)(v10 + 172) & 2) != 0 )
    {
      a2 = 2439;
      result = (unsigned __int8 *)sub_67DDB0(v16, 2439, (_QWORD *)(v10 + 64));
      goto LABEL_3;
    }
    a2 = 547;
    if ( v11 )
    {
      a2 = 1033;
      if ( *(_BYTE *)(v11 + 40) != 22 )
        a2 = 547;
    }
    result = (unsigned __int8 *)sub_67E1D0(v16, a2, *(_QWORD *)v10);
    if ( (__int64 *)v6 == v5 )
      return result;
LABEL_4:
    if ( *((_BYTE *)v5 + 32) )
      goto LABEL_7;
    v5 = (__int64 *)v5[5];
LABEL_6:
    if ( (__int64 *)v6 == v5 )
      return result;
LABEL_7:
    v5 = (__int64 *)*v5;
  }
}

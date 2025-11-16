// Function: sub_7F7930
// Address: 0x7f7930
//
__int64 *__fastcall sub_7F7930(__int64 a1, int a2, char *a3, __int64 a4, int *a5, __int64 a6, int a7)
{
  __int64 v12; // rax
  __m128i *v13; // r15
  __int16 v14; // r10
  __int64 *v15; // r15
  size_t v17; // rax
  void *v18; // rax
  size_t v19; // rdi
  __int64 v20; // rax
  size_t v21; // rax
  void *v22; // rax
  char *v23; // rax
  size_t v24; // r15
  char *v25; // r15
  size_t v26; // rax
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-98h]
  char *s; // [rsp+20h] [rbp-80h]
  size_t n; // [rsp+28h] [rbp-78h]
  char *na; // [rsp+28h] [rbp-78h]
  char *nb; // [rsp+28h] [rbp-78h]
  char v33[112]; // [rsp+30h] [rbp-70h] BYREF

  if ( !a3 || a7 )
  {
    v12 = sub_72CBE0();
    v13 = sub_7F7840(a3, 2, v12, a1);
    sub_7362F0((__int64)v13, 0);
    sub_7604D0((__int64)v13, 0xBu);
    sub_7605A0((__int64)v13);
    v14 = a2;
    if ( !a2 )
      goto LABEL_5;
    goto LABEL_4;
  }
  s = (char *)sub_7217A0();
  n = strlen(a3);
  v17 = strlen(s);
  if ( a2 )
  {
    v28 = n + v17 + 1;
    sprintf(v33, "__prio%d", a2);
    v21 = strlen(v33);
    v22 = (void *)sub_7E1510(v28 + v21);
    v23 = (char *)memcpy(v22, a3, n);
    v24 = n;
    nb = v23;
    v25 = &v23[v24];
    strcpy(v25, s);
    v26 = strlen(s);
    strcpy(&v25[v26], v33);
    v27 = sub_72CBE0();
    v13 = sub_7F7840(nb, 2, v27, a1);
    sub_7362F0((__int64)v13, 0);
    sub_7604D0((__int64)v13, 0xBu);
    sub_7605A0((__int64)v13);
    v14 = a2;
LABEL_4:
    v13[22].m128i_i16[0] = v14;
    goto LABEL_5;
  }
  v18 = (void *)sub_7E1510(n + v17 + 1);
  v19 = n;
  na = (char *)memcpy(v18, a3, n);
  strcpy(&na[v19], s);
  v20 = sub_72CBE0();
  v13 = sub_7F7840(na, 2, v20, a1);
  sub_7362F0((__int64)v13, 0);
  sub_7604D0((__int64)v13, 0xBu);
  sub_7605A0((__int64)v13);
LABEL_5:
  v15 = sub_7F54F0((__int64)v13, 1, 0, a5);
  sub_7F6C60((__int64)v15, *a5, a6);
  if ( a1 )
    v15[5] = (__int64)sub_7E2270(a1);
  sub_7E1740(v15[10], a4);
  sub_7E17A0(*(_QWORD *)(v15[10] + 72));
  return v15;
}

// Function: sub_3118A60
// Address: 0x3118a60
//
void __fastcall sub_3118A60(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r14
  char *v8; // r13
  __int64 v9; // rbx
  __int64 v11; // rax
  char *v12; // r10
  char *v13; // r11
  char *v14; // r11
  __int64 v15; // rax
  size_t v16; // rbx
  size_t v17; // r13
  __int64 *v18; // r14
  char *v19; // r11
  size_t v20; // rdx
  int v21; // eax
  int v22; // r12d
  __int64 v23; // rbx
  __int64 v24; // rax
  char *v26; // [rsp+8h] [rbp-B8h]
  char *v27; // [rsp+10h] [rbp-B0h]
  char *v28; // [rsp+18h] [rbp-A8h]
  __int64 v29; // [rsp+20h] [rbp-A0h]
  __int64 v30; // [rsp+28h] [rbp-98h]
  char *v31; // [rsp+28h] [rbp-98h]
  char *v32; // [rsp+28h] [rbp-98h]
  char *v33; // [rsp+28h] [rbp-98h]
  void *s1; // [rsp+30h] [rbp-90h] BYREF
  size_t n; // [rsp+38h] [rbp-88h]
  __int64 v36; // [rsp+40h] [rbp-80h] BYREF
  char v37; // [rsp+50h] [rbp-70h]
  void *s2; // [rsp+60h] [rbp-60h] BYREF
  size_t v39; // [rsp+68h] [rbp-58h]
  __int64 v40; // [rsp+70h] [rbp-50h] BYREF
  char v41; // [rsp+80h] [rbp-40h]

  if ( a4 )
  {
    v6 = a5;
    if ( a5 )
    {
      v7 = (__int64)a1;
      v8 = a2;
      v9 = a4;
      if ( a4 + a5 == 2 )
      {
        v27 = a1;
        v14 = a2;
LABEL_12:
        v31 = v14;
        sub_31185E0((__int64)&s2, a6, *(_DWORD *)(*(_QWORD *)v27 + 12LL));
        sub_31185E0((__int64)&s1, a6, *(_DWORD *)(*(_QWORD *)v31 + 12LL));
        v16 = n;
        v17 = v39;
        v18 = (__int64 *)s1;
        v19 = v31;
        v20 = v39;
        if ( n <= v39 )
          v20 = n;
        if ( !v20 || (v21 = memcmp(s1, s2, v20), v19 = v31, (v22 = v21) == 0) )
        {
          v23 = v16 - v17;
          v22 = 0x7FFFFFFF;
          if ( v23 <= 0x7FFFFFFF )
          {
            v22 = 0x80000000;
            if ( v23 >= (__int64)0xFFFFFFFF80000000LL )
              v22 = v23;
          }
        }
        if ( v37 )
        {
          v37 = 0;
          if ( v18 != &v36 )
          {
            v33 = v19;
            j_j___libc_free_0((unsigned __int64)v18);
            v19 = v33;
          }
        }
        if ( v41 )
        {
          v41 = 0;
          if ( s2 != &v40 )
          {
            v32 = v19;
            j_j___libc_free_0((unsigned __int64)s2);
            v19 = v32;
          }
        }
        if ( v22 < 0 )
        {
          v24 = *(_QWORD *)v27;
          *(_QWORD *)v27 = *(_QWORD *)v19;
          *(_QWORD *)v19 = v24;
        }
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v29 = v9 / 2;
        v11 = sub_31186C0((__int64)v8, a3, v7 + 8 * (v9 / 2), a6);
        v12 = (char *)(v7 + 8 * (v9 / 2));
        v13 = (char *)v11;
        v30 = (v11 - (__int64)v8) >> 3;
        while ( 1 )
        {
          v26 = v13;
          v28 = v12;
          v27 = sub_3117E70(v12, v8, v13);
          sub_3118A60(v7, v28, v27, v29, v30, a6);
          v6 -= v30;
          v9 -= v29;
          if ( !v9 )
            break;
          v14 = v26;
          if ( !v6 )
            break;
          if ( v6 + v9 == 2 )
            goto LABEL_12;
          v7 = (__int64)v27;
          v8 = v26;
          if ( v6 < v9 )
            goto LABEL_5;
LABEL_10:
          v30 = v6 / 2;
          v15 = sub_3118890(v7, (__int64)v8, (__int64)&v8[8 * (v6 / 2)], a6);
          v13 = &v8[8 * (v6 / 2)];
          v12 = (char *)v15;
          v29 = (v15 - v7) >> 3;
        }
      }
    }
  }
}

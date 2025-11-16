// Function: sub_3724400
// Address: 0x3724400
//
void **__fastcall sub_3724400(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, void **a6, __int64 a7)
{
  void **v7; // r12
  __int64 v8; // rax
  void **v9; // r14
  size_t v10; // rbx
  void **result; // rax
  void ***v12; // r13
  void *v13; // r15
  unsigned __int64 v14; // rbx
  __int64 v15; // rbx
  __int64 v16; // r13
  char *v17; // r13
  void ***v18; // r15
  void ***v19; // rbx
  char *v20; // r13
  void **v21; // r12
  unsigned __int64 v22; // r14
  char *v23; // r13
  char *v24; // rax
  __int64 v25; // r10
  __int64 v26; // rcx
  char *v27; // r11
  __int64 v28; // r8
  __int64 v29; // r14
  char *v30; // r15
  char *v31; // rax
  int v32; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+18h] [rbp-58h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+20h] [rbp-50h]
  __int64 v36; // [rsp+20h] [rbp-50h]
  __int64 *v37; // [rsp+20h] [rbp-50h]
  char *v38; // [rsp+30h] [rbp-40h]
  __int64 v39; // [rsp+30h] [rbp-40h]
  char *v40; // [rsp+38h] [rbp-38h]
  char *v41; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    v7 = a6;
    v40 = (char *)a3;
    v8 = a7;
    if ( a5 <= a7 )
      v8 = a5;
    if ( v8 >= a4 )
    {
      v9 = (void **)a2;
      v10 = a2 - a1;
      if ( a2 != a1 )
        memmove(a6, a1, v10);
      result = (void **)((char *)v7 + v10);
      v12 = (void ***)a1;
      v38 = (char *)v7 + v10;
      if ( v7 != (void **)((char *)v7 + v10) )
      {
        while ( v40 != (char *)v9 )
        {
          v13 = *v7;
          v14 = (*(__int64 (__fastcall **)(void *))(*(_QWORD *)*v9 + 16LL))(*v9);
          if ( v14 < (*(__int64 (__fastcall **)(void *))(*(_QWORD *)v13 + 16LL))(v13) )
            result = (void **)*v9++;
          else
            result = (void **)*v7++;
          *v12++ = result;
          if ( v38 == (char *)v7 )
            return result;
        }
        return (void **)memmove(v12, v7, v38 - (char *)v7);
      }
      return result;
    }
    v15 = a5;
    if ( a5 <= a7 )
      break;
    v33 = a4;
    if ( a5 < a4 )
    {
      v39 = a4 / 2;
      v37 = (__int64 *)&a1[8 * (a4 / 2)];
      v31 = (char *)sub_3722460(a2, a3, v37);
      v27 = (char *)v37;
      v26 = v33;
      v25 = a7;
      v23 = v31;
      v28 = (v31 - a2) >> 3;
    }
    else
    {
      v35 = a5 / 2;
      v23 = &a2[8 * (a5 / 2)];
      v24 = (char *)sub_37224F0(a1, (__int64)a2, v23);
      v25 = a7;
      v26 = v33;
      v27 = v24;
      v28 = v35;
      v39 = (v24 - a1) >> 3;
    }
    v29 = v26 - v39;
    v34 = v25;
    v36 = v28;
    v32 = (int)v27;
    v30 = sub_37242E0(v27, a2, v23, v26 - v39, v28, v7, v25);
    sub_3724400((_DWORD)a1, v32, (_DWORD)v30, v39, v36, (_DWORD)v7, v34);
    a6 = v7;
    a4 = v29;
    a2 = v23;
    a7 = v34;
    a3 = (__int64)v40;
    a5 = v15 - v36;
    a1 = v30;
  }
  v16 = a3 - (_QWORD)a2;
  if ( (char *)a3 != a2 )
    memmove(a6, a2, a3 - (_QWORD)a2);
  result = (void **)((char *)v7 + v16);
  if ( a2 != a1 )
  {
    if ( v7 == result )
      return result;
    v17 = v40;
    v41 = (char *)v7;
    v18 = (void ***)(a2 - 8);
    v19 = (void ***)(v17 - 8);
    v20 = (char *)(result - 1);
    while ( 1 )
    {
      v21 = *v18;
      v22 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)v20 + 16LL))(*(_QWORD *)v20);
      if ( v22 < (*((__int64 (__fastcall **)(void **))*v21 + 2))(v21) )
      {
        result = *v18;
        *v19 = *v18;
        if ( a1 == (char *)v18 )
        {
          if ( v41 != v20 + 8 )
            return (void **)memmove((char *)v19 - (v20 + 8 - v41), v41, v20 + 8 - v41);
          return result;
        }
        --v18;
      }
      else
      {
        result = *(void ***)v20;
        *v19 = *(void ***)v20;
        if ( v41 == v20 )
          return result;
        v20 -= 8;
      }
      --v19;
    }
  }
  if ( v7 != result )
    return (void **)memmove(a2, v7, v40 - a2);
  return result;
}

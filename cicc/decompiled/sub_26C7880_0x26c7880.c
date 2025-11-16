// Function: sub_26C7880
// Address: 0x26c7880
//
__int64 __fastcall sub_26C7880(_QWORD *a1, int *a2, size_t a3)
{
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 result; // rax
  __int64 v8; // rcx
  __int64 *v9; // rax
  size_t v10; // rsi
  unsigned __int64 v11; // r8
  _QWORD *v12; // r11
  _QWORD *v13; // rax
  _QWORD *v14; // rdi
  _QWORD *v15; // rax
  size_t v16; // rax
  __int64 *v17; // rax
  _QWORD *v18; // [rsp+8h] [rbp-78h]
  int *v19; // [rsp+10h] [rbp-70h]
  size_t v20; // [rsp+18h] [rbp-68h]
  size_t v21; // [rsp+28h] [rbp-58h] BYREF
  int *v22[2]; // [rsp+30h] [rbp-50h] BYREF
  char v23; // [rsp+40h] [rbp-40h]

  v19 = a2;
  v20 = a3;
  v22[0] = (int *)sub_26BA4C0(a2, a3);
  v5 = sub_C1DD00(a1 + 1, (unsigned __int64)v22[0] % a1[2], v22, (__int64)v22[0]);
  if ( v5 )
  {
    v6 = *v5;
    if ( v6 )
      return v6 + 16;
  }
  v8 = a1[12];
  if ( v8 )
  {
    if ( *(_QWORD *)(v8 + 24) )
    {
      v18 = (_QWORD *)a1[12];
      v10 = sub_26BA4C0(a2, a3);
      v11 = v18[1];
      v12 = *(_QWORD **)(*v18 + 8 * (v10 % v11));
      if ( v12 )
      {
        v13 = (_QWORD *)*v12;
        if ( *(_QWORD *)(*v12 + 8LL) == v10 )
        {
LABEL_17:
          v15 = (_QWORD *)*v12;
          if ( *v12 )
          {
            v19 = (int *)v15[2];
            if ( v19 )
            {
              v20 = v15[3];
              v16 = sub_26BA4C0(v19, v20);
            }
            else
            {
              v20 = 0;
              v16 = sub_26BA4C0(0, 0);
            }
            v22[0] = (int *)v16;
            v17 = sub_C1DD00(a1 + 1, v16 % a1[2], v22, v16);
            if ( v17 )
            {
              v6 = *v17;
              if ( v6 )
                return v6 + 16;
            }
          }
        }
        else
        {
          while ( 1 )
          {
            v14 = (_QWORD *)*v13;
            if ( !*v13 )
              break;
            v12 = v13;
            if ( v10 % v11 != v14[1] % v11 )
              break;
            v13 = (_QWORD *)*v13;
            if ( v14[1] == v10 )
              goto LABEL_17;
          }
        }
      }
    }
  }
  result = a1[11];
  if ( result )
  {
    sub_C21B60((__int64)v22, a1[11], (__int64)v19, v20);
    if ( !v23 )
      return 0;
    v21 = sub_26BA4C0(v22[0], (size_t)v22[1]);
    v9 = sub_C1DD00(a1 + 1, v21 % a1[2], &v21, v21);
    if ( !v9 )
      return 0;
    v6 = *v9;
    if ( !v6 )
      return 0;
    return v6 + 16;
  }
  return result;
}

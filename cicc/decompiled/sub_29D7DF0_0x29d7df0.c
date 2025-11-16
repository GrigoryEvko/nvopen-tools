// Function: sub_29D7DF0
// Address: 0x29d7df0
//
__int64 __fastcall sub_29D7DF0(__int64 a1, __int64 *a2, __int64 *a3)
{
  unsigned int *v4; // r13
  unsigned int *v5; // r15
  unsigned int v6; // ebx
  unsigned int v7; // eax
  __int64 result; // rax
  int v9; // ebx
  int v10; // eax
  int v11; // ebx
  int v12; // eax
  unsigned int v13; // ebx
  unsigned int v14; // eax
  void *v15; // rbx
  unsigned int v16; // [rsp+8h] [rbp-58h]
  unsigned int v17; // [rsp+8h] [rbp-58h]
  unsigned __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-48h]
  unsigned __int64 v20; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-38h]

  v4 = (unsigned int *)*a3;
  v5 = (unsigned int *)*a2;
  v6 = sub_C336A0(*a3);
  v7 = sub_C336A0((__int64)v5);
  result = sub_29D7CF0(a1, v7, v6);
  if ( !(_DWORD)result )
  {
    v9 = sub_C336B0(v4);
    v10 = sub_C336B0(v5);
    result = sub_29D7CF0(a1, v10, v9);
    if ( !(_DWORD)result )
    {
      v11 = sub_C336C0((__int64)v4);
      v12 = sub_C336C0((__int64)v5);
      result = sub_29D7CF0(a1, v12, v11);
      if ( !(_DWORD)result )
      {
        v13 = sub_C336D0((__int64)v4);
        v14 = sub_C336D0((__int64)v5);
        result = sub_29D7CF0(a1, v14, v13);
        if ( !(_DWORD)result )
        {
          v15 = sub_C33340();
          if ( (void *)*a3 == v15 )
            sub_C3E660((__int64)&v20, (__int64)a3);
          else
            sub_C3A850((__int64)&v20, a3);
          if ( (void *)*a2 == v15 )
            sub_C3E660((__int64)&v18, (__int64)a2);
          else
            sub_C3A850((__int64)&v18, a2);
          result = sub_29D7D50(a1, (__int64)&v18, (__int64)&v20);
          if ( v19 > 0x40 && v18 )
          {
            v16 = result;
            j_j___libc_free_0_0(v18);
            result = v16;
          }
          if ( v21 > 0x40 )
          {
            if ( v20 )
            {
              v17 = result;
              j_j___libc_free_0_0(v20);
              return v17;
            }
          }
        }
      }
    }
  }
  return result;
}

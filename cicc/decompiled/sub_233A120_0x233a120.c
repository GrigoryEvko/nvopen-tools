// Function: sub_233A120
// Address: 0x233a120
//
__int64 __fastcall sub_233A120(
        __int64 a1,
        void *a2,
        unsigned __int64 a3,
        const void *a4,
        size_t a5,
        __int64 a6,
        char a7)
{
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdx
  void *v12; // rdi
  unsigned __int64 v13; // rcx
  char *v14; // rdx
  unsigned int v15; // eax
  unsigned int v16; // ebx
  __int64 v17; // rdx
  __int64 v18; // r15
  __int64 v19; // rax
  int v21; // eax
  char v22; // dl
  char v23; // al
  unsigned __int64 v24; // [rsp+8h] [rbp-E8h]
  void *s1; // [rsp+10h] [rbp-E0h] BYREF
  unsigned __int64 v26; // [rsp+18h] [rbp-D8h]
  __int64 v27; // [rsp+28h] [rbp-C8h] BYREF
  void *v28; // [rsp+30h] [rbp-C0h] BYREF
  unsigned __int64 v29; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v30[4]; // [rsp+40h] [rbp-B0h] BYREF
  _QWORD v31[4]; // [rsp+60h] [rbp-90h] BYREF
  char v32; // [rsp+80h] [rbp-70h]
  _QWORD v33[2]; // [rsp+88h] [rbp-68h] BYREF
  _QWORD v34[2]; // [rsp+98h] [rbp-58h] BYREF
  _QWORD v35[9]; // [rsp+A8h] [rbp-48h] BYREF

  s1 = a2;
  v26 = a3;
  if ( a3 )
  {
    do
    {
      v28 = 0;
      v29 = 0;
      LOBYTE(v31[0]) = 59;
      v10 = sub_C931B0((__int64 *)&s1, v31, 1u, 0);
      if ( v10 == -1 )
      {
        v12 = s1;
        v10 = v26;
        v13 = 0;
        v14 = 0;
      }
      else
      {
        v11 = v10 + 1;
        v12 = s1;
        if ( v10 + 1 > v26 )
        {
          v11 = v26;
          v13 = 0;
        }
        else
        {
          v13 = v26 - v11;
        }
        v14 = (char *)s1 + v11;
        if ( v10 > v26 )
          v10 = v26;
      }
      v28 = v12;
      v29 = v10;
      s1 = v14;
      v26 = v13;
      if ( v10 == a5 )
      {
        if ( !a5 )
          continue;
        v24 = v13;
        v21 = memcmp(v12, a4, a5);
        v13 = v24;
        if ( !v21 )
          continue;
      }
      v15 = sub_C63BB0();
      v32 = 1;
      v16 = v15;
      v18 = v17;
      v33[1] = &a7;
      v31[0] = "invalid {1} pass parameter '{0}' ";
      v31[2] = v35;
      v31[1] = 33;
      v31[3] = 2;
      v33[0] = &unk_49DB108;
      v34[0] = &unk_49DB108;
      v34[1] = &v28;
      v35[0] = v34;
      v35[1] = v33;
      sub_23328D0((__int64)v30, (__int64)v31);
      sub_23058C0(&v27, (__int64)v30, v16, v18);
      v19 = v27;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v19 & 0xFFFFFFFFFFFFFFFELL;
      sub_2240A30(v30);
      return a1;
    }
    while ( v13 );
    v22 = 1;
  }
  else
  {
    v22 = 0;
  }
  v23 = *(_BYTE *)(a1 + 8);
  *(_BYTE *)a1 = v22;
  *(_BYTE *)(a1 + 8) = v23 & 0xFC | 2;
  return a1;
}

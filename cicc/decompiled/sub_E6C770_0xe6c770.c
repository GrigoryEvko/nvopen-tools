// Function: sub_E6C770
// Address: 0xe6c770
//
__int64 __fastcall sub_E6C770(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, const char *a5, __int64 a6)
{
  __int64 v6; // rax
  const char *v7; // rdx
  const char *v8; // rcx
  char v9; // al
  char v10; // al
  const char *v12; // [rsp+0h] [rbp-60h] BYREF
  const char *v13; // [rsp+8h] [rbp-58h]
  __int64 *v14; // [rsp+10h] [rbp-50h]
  __int64 v15; // [rsp+18h] [rbp-48h]
  __int16 v16; // [rsp+20h] [rbp-40h]
  const char *v17[4]; // [rsp+30h] [rbp-30h] BYREF
  __int16 v18; // [rsp+50h] [rbp-10h]

  v6 = *(_QWORD *)(a1 + 152);
  v7 = *(const char **)(v6 + 88);
  v8 = *(const char **)(v6 + 96);
  v9 = *((_BYTE *)a2 + 32);
  if ( v9 )
  {
    if ( v9 == 1 )
    {
      v12 = v7;
      a5 = v8;
      v16 = 261;
      v10 = 5;
      v13 = v8;
    }
    else
    {
      if ( *((_BYTE *)a2 + 33) == 1 )
      {
        a6 = a2[1];
        a2 = (__int64 *)*a2;
      }
      else
      {
        v9 = 2;
      }
      v13 = v8;
      v14 = a2;
      v15 = a6;
      LOBYTE(v16) = 5;
      v12 = v7;
      v7 = (const char *)&v12;
      HIBYTE(v16) = v9;
      v10 = 2;
    }
    v17[0] = v7;
    v17[1] = a5;
    v17[2] = "$parent_frame_offset";
    LOBYTE(v18) = v10;
    HIBYTE(v18) = 3;
    return sub_E6C460(a1, v17);
  }
  else
  {
    v16 = 256;
    v18 = 256;
    return sub_E6C460(a1, v17);
  }
}

// Function: sub_E6C650
// Address: 0xe6c650
//
__int64 __fastcall sub_E6C650(__int64 a1, __int64 *a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  _QWORD *v7; // rcx
  __int64 v8; // r10
  char v9; // al
  char v10; // al
  _QWORD *v12; // [rsp+0h] [rbp-90h] BYREF
  __int64 v13; // [rsp+8h] [rbp-88h]
  __int64 *v14; // [rsp+10h] [rbp-80h]
  __int64 v15; // [rsp+18h] [rbp-78h]
  __int16 v16; // [rsp+20h] [rbp-70h]
  _QWORD v17[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v18; // [rsp+50h] [rbp-40h]
  const char *v19[2]; // [rsp+60h] [rbp-30h] BYREF
  int v20; // [rsp+70h] [rbp-20h]
  __int16 v21; // [rsp+80h] [rbp-10h]

  v6 = *(_QWORD *)(a1 + 152);
  v7 = *(_QWORD **)(v6 + 88);
  v8 = *(_QWORD *)(v6 + 96);
  v9 = *((_BYTE *)a2 + 32);
  if ( v9 )
  {
    if ( v9 == 1 )
    {
      v12 = v7;
      a5 = v8;
      v10 = 5;
      v13 = v8;
      v16 = 261;
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
      v7 = &v12;
      HIBYTE(v16) = v9;
      v10 = 2;
    }
    LOBYTE(v18) = v10;
    v17[0] = v7;
    v19[0] = (const char *)v17;
    v17[1] = a5;
    v17[2] = "$frame_escape_";
    HIBYTE(v18) = 3;
    v19[1] = 0;
    v20 = a3;
    v21 = 2306;
    return sub_E6C460(a1, v19);
  }
  else
  {
    v16 = 256;
    v18 = 256;
    v21 = 256;
    return sub_E6C460(a1, v19);
  }
}

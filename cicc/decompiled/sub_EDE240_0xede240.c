// Function: sub_EDE240
// Address: 0xede240
//
_QWORD *__fastcall sub_EDE240(_QWORD *a1, unsigned __int64 *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 *v4; // rcx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdi
  _QWORD *v8; // rcx
  __int64 v10; // rax
  __int64 v11; // rbx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // [rsp+8h] [rbp-C8h] BYREF
  __int64 *v14; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v15; // [rsp+30h] [rbp-A0h]
  __int64 v16[4]; // [rsp+40h] [rbp-90h] BYREF
  char v17; // [rsp+60h] [rbp-70h]
  _QWORD v18[2]; // [rsp+68h] [rbp-68h] BYREF
  _QWORD v19[2]; // [rsp+78h] [rbp-58h] BYREF
  _QWORD v20[2]; // [rsp+88h] [rbp-48h] BYREF
  _QWORD v21[7]; // [rsp+98h] [rbp-38h] BYREF

  v4 = (unsigned __int64 *)(a3 + a4);
  v6 = *v4;
  v7 = *v4 - 2;
  v13 = *v4;
  if ( v7 > 1 )
  {
    v16[1] = 79;
    v16[0] = (__int64)"MemProf version {} not supported; requires version between {} and {}, inclusive";
    v16[2] = (__int64)v21;
    v18[1] = &unk_3F87940;
    v19[1] = &unk_3F87948;
    v18[0] = &unk_49E4CE8;
    v19[0] = &unk_49E4CE8;
    v20[0] = &unk_49E4CE8;
    v20[1] = &v13;
    v21[0] = v20;
    v21[1] = v19;
    v21[2] = v18;
    v16[3] = 3;
    v17 = 1;
    v15 = 263;
    v14 = v16;
    v10 = sub_22077B0(48);
    v11 = v10;
    if ( v10 )
    {
      *(_DWORD *)(v10 + 8) = 5;
      *(_QWORD *)v10 = &unk_49E4BC8;
      sub_CA0F50((__int64 *)(v10 + 16), (void **)&v14);
    }
    *a1 = v11 | 1;
  }
  else
  {
    *a2 = v6;
    v8 = v4 + 1;
    if ( v6 == 3 )
    {
      sub_EDDF80(v16, (__int64)a2, a3, (__int64)v8);
      v12 = v16[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v16[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v16[0] = 0;
        *a1 = v12 | 1;
        sub_9C66B0(v16);
        return a1;
      }
      v16[0] = 0;
      sub_9C66B0(v16);
    }
    else
    {
      sub_EDDB80(v16, a2, a3, v8);
      if ( (v16[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        *a1 = v16[0] & 0xFFFFFFFFFFFFFFFELL | 1;
        return a1;
      }
    }
    *a1 = 1;
  }
  return a1;
}

// Function: sub_23B2C50
// Address: 0x23b2c50
//
__int64 __fastcall sub_23B2C50(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  void **v5; // rax
  void **v6; // rdx
  __int64 **v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rcx

  if ( *(_BYTE *)(a3 + 76) )
  {
    v5 = *(void ***)(a3 + 56);
    v6 = &v5[*(unsigned int *)(a3 + 68)];
    if ( v5 != v6 )
    {
      a4 = (__int64 *)&unk_4FDE338;
      while ( *v5 != &unk_4FDE338 )
      {
        if ( v6 == ++v5 )
          goto LABEL_8;
      }
      return 1;
    }
  }
  else if ( sub_C8CA60(a3 + 48, (__int64)&unk_4FDE338) )
  {
    return 1;
  }
LABEL_8:
  if ( *(_BYTE *)(a3 + 28) )
  {
    v8 = *(__int64 ***)(a3 + 8);
    v9 = (__int64)&v8[*(unsigned int *)(a3 + 20)];
    if ( v8 != (__int64 **)v9 )
    {
      a4 = &qword_4F82400;
      while ( *v8 != &qword_4F82400 )
      {
        if ( (__int64 **)v9 == ++v8 )
          goto LABEL_15;
      }
      return 0;
    }
  }
  else if ( sub_C8CA60(a3, (__int64)&qword_4F82400) )
  {
    return 0;
  }
LABEL_15:
  if ( (unsigned __int8)sub_B19060(a3, (__int64)&unk_4FDE338, v9, (__int64)a4)
    || (unsigned __int8)sub_B19060(a3, (__int64)&qword_4F82400, v10, v11)
    || (unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82420, v12, v13)
    || (unsigned __int8)sub_B19060(a3, (__int64)&qword_4F82400, v14, v15) )
  {
    return 0;
  }
  return (unsigned int)sub_B19060(a3, (__int64)&unk_4F82408, v16, v17) ^ 1;
}

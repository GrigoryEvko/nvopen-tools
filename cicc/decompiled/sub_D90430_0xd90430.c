// Function: sub_D90430
// Address: 0xd90430
//
__int64 __fastcall sub_D90430(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v4; // rdi
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  unsigned int v8; // r8d

  v2 = sub_D8E7E0(a1, a2);
  v3 = *(unsigned __int8 *)(v2 + 76);
  v4 = v2;
  if ( (_BYTE)v3 )
  {
    v5 = *(_QWORD **)(v2 + 56);
    v6 = &v5[*(unsigned int *)(v4 + 68)];
    if ( v5 == v6 )
    {
      return 0;
    }
    else
    {
      while ( a2 != *v5 )
      {
        if ( v6 == ++v5 )
          return 0;
      }
      return v3;
    }
  }
  else
  {
    LOBYTE(v8) = sub_C8CA60(v2 + 48, a2) != 0;
    return v8;
  }
}

// Function: sub_2EC4810
// Address: 0x2ec4810
//
__int64 __fastcall sub_2EC4810(_QWORD *a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // rdi

  v2 = (_QWORD *)a1[436];
  v3 = (_QWORD *)a1[435];
  *a1 = &unk_4A29DA8;
  if ( v2 != v3 )
  {
    do
    {
      if ( *v3 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v3 + 16LL))(*v3);
      ++v3;
    }
    while ( v2 != v3 );
    v3 = (_QWORD *)a1[435];
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
  v4 = a1[434];
  if ( v4 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 16LL))(v4);
  return sub_2EC45A0((__int64)a1);
}

// Function: sub_317ED60
// Address: 0x317ed60
//
__int64 __fastcall sub_317ED60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int128 v8; // rdi
  __int64 v9; // r12
  __int64 v10; // rdx
  int *v11; // r8
  size_t v12; // rcx
  _QWORD *v13; // rdi
  __int64 v15; // rax

  v7 = sub_B10CD0(a2 + 48);
  if ( !v7 )
    return 0;
  *(_QWORD *)&v8 = a3;
  *((_QWORD *)&v8 + 1) = a4;
  v9 = v7;
  v11 = (int *)sub_C16140(v8, (__int64)"selected", 8);
  v12 = v10;
  if ( v10 )
  {
    if ( unk_4F838D1 )
    {
      v15 = sub_B2F650((__int64)v11, v10);
      v11 = 0;
      v12 = v15;
    }
  }
  v13 = sub_317ED00(a1, v9, v11, v12);
  if ( v13 )
    return sub_317E470((__int64)v13);
  else
    return 0;
}

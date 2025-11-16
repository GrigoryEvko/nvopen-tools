// Function: sub_2A634B0
// Address: 0x2a634b0
//
__int64 __fastcall sub_2A634B0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // [rsp-28h] [rbp-28h]
  __int64 v12; // [rsp-28h] [rbp-28h]

  v6 = *a2;
  if ( (_BYTE)v6 == 6 )
    return 0;
  if ( (unsigned int)(v6 - 4) <= 1 )
  {
    if ( *((_DWORD *)a2 + 8) > 0x40u )
    {
      v9 = *((_QWORD *)a2 + 3);
      if ( v9 )
      {
        v11 = a3;
        j_j___libc_free_0_0(v9);
        a3 = v11;
      }
    }
    if ( *((_DWORD *)a2 + 4) > 0x40u )
    {
      v10 = *((_QWORD *)a2 + 1);
      if ( v10 )
      {
        v12 = a3;
        j_j___libc_free_0_0(v10);
        a3 = v12;
      }
    }
  }
  *a2 = 6;
  sub_2A62F90(a1, a2, a3, a4, a5, a6);
  return 1;
}

// Function: sub_10A5170
// Address: 0x10a5170
//
__int64 __fastcall sub_10A5170(__int64 a1, _BYTE *a2)
{
  __int64 result; // rax
  unsigned __int8 *v4; // r12
  int v5; // edx
  int v6; // edx
  __int64 v7; // rdi
  unsigned __int8 *v8; // rdx
  char v9; // dl
  char v10; // dl
  unsigned __int8 *v11; // r12
  __int64 v12; // [rsp-28h] [rbp-28h]
  char v13; // [rsp-20h] [rbp-20h]

  result = 0;
  if ( *a2 == 68 )
  {
    v4 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    v5 = *v4;
    if ( (unsigned __int8)v5 > 0x1Cu )
    {
      v6 = v5 - 29;
    }
    else
    {
      if ( (_BYTE)v5 != 5 )
        return result;
      v6 = *((unsigned __int16 *)v4 + 1);
    }
    if ( v6 == 47
      && ((v7 = *(_QWORD *)a1, (v4[7] & 0x40) == 0)
        ? (v8 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)])
        : (v8 = (unsigned __int8 *)*((_QWORD *)v4 - 1)),
          (v12 = sub_9208B0(v7, *(_QWORD *)(*(_QWORD *)v8 + 8LL)),
           v13 = v9,
           sub_9208B0(*(_QWORD *)a1, *((_QWORD *)v4 + 1)) == v12)
       && v10 == v13
       && ((v4[7] & 0x40) == 0
         ? (v11 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)])
         : (v11 = (unsigned __int8 *)*((_QWORD *)v4 - 1)),
           *(_QWORD *)v11)) )
    {
      **(_QWORD **)(a1 + 8) = *(_QWORD *)v11;
      return 1;
    }
    else
    {
      return 0;
    }
  }
  return result;
}

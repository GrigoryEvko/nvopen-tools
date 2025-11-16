// Function: sub_2D57BD0
// Address: 0x2d57bd0
//
__int64 __fastcall sub_2D57BD0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r12

  for ( result = *((unsigned int *)a1 + 2); (_DWORD)result; result = *((unsigned int *)a1 + 2) )
  {
    result = *a1 + 8 * result - 8;
    v7 = *(_QWORD *)result;
    if ( *(_QWORD *)result == a2 )
      break;
    *(_QWORD *)result = 0;
    v4 = *a1;
    v5 = (unsigned int)(*((_DWORD *)a1 + 2) - 1);
    *((_DWORD *)a1 + 2) = v5;
    v6 = *(_QWORD *)(v4 + 8 * v5);
    if ( v6 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 16LL))(v7);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
  }
  return result;
}

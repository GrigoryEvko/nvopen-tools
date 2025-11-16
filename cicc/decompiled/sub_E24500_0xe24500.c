// Function: sub_E24500
// Address: 0xe24500
//
__int64 __fastcall sub_E24500(__int64 **a1, __int64 *a2, char *a3)
{
  __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  __int64 result; // rax
  __int64 v7; // rcx
  char v8; // dl
  __int64 *v9; // rax
  __int64 *v10; // r12
  __int64 *v11; // rdx

  v4 = **a1;
  v5 = (v4 + (*a1)[1] + 7) & 0xFFFFFFFFFFFFFFF8LL;
  (*a1)[1] = v5 - v4 + 32;
  if ( (*a1)[1] <= (unsigned __int64)(*a1)[2] )
  {
    result = 0;
    if ( !v5 )
      return result;
    result = v5;
    goto LABEL_4;
  }
  v9 = (__int64 *)sub_22077B0(32);
  v10 = v9;
  if ( v9 )
  {
    *v9 = 0;
    v9[1] = 0;
    v9[2] = 0;
    v9[3] = 0;
  }
  result = sub_2207820(4096);
  v11 = *a1;
  *a1 = v10;
  *v10 = result;
  v10[3] = (__int64)v11;
  v10[2] = 4096;
  v10[1] = 32;
  if ( result )
  {
LABEL_4:
    v7 = *a2;
    v8 = *a3;
    *(_DWORD *)(result + 8) = 23;
    *(_QWORD *)(result + 16) = v7;
    *(_QWORD *)result = &unk_49E0F10;
    *(_BYTE *)(result + 24) = v8;
  }
  return result;
}

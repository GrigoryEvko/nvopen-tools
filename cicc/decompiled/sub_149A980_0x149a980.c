// Function: sub_149A980
// Address: 0x149a980
//
__int64 __fastcall sub_149A980(_QWORD *a1, void *a2, void *a3, size_t a4, void *a5, size_t a6)
{
  _QWORD *v6; // r13
  __int64 v7; // rcx
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 result; // rax
  int v12; // eax
  int v13; // eax
  int v14; // [rsp+4h] [rbp-8Ch]
  __int64 v15; // [rsp+8h] [rbp-88h]
  void *s2; // [rsp+10h] [rbp-80h] BYREF
  size_t n; // [rsp+18h] [rbp-78h]
  _QWORD v18[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v19; // [rsp+30h] [rbp-60h]
  _QWORD v20[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v21; // [rsp+50h] [rbp-40h]

  v6 = a1;
  v14 = (int)a2;
  if ( *(_QWORD *)(a1[24] + 32LL) )
  {
    s2 = a5;
    n = a6;
  }
  else
  {
    s2 = a3;
    n = a4;
  }
  v7 = *((unsigned int *)a1 + 52);
  if ( *((_DWORD *)a1 + 52) )
  {
    v8 = 0;
    v9 = a1[25];
    while ( 1 )
    {
      if ( *(_QWORD *)(v9 + 8) == n )
      {
        if ( !n )
          break;
        a1 = *(_QWORD **)v9;
        a2 = s2;
        v15 = v7;
        v13 = memcmp(*(const void **)v9, s2, n);
        v7 = v15;
        if ( !v13 )
          break;
      }
      ++v8;
      v9 += 48;
      if ( v7 == v8 )
        goto LABEL_7;
    }
    v12 = *(_DWORD *)(v9 + 40);
  }
  else
  {
LABEL_7:
    v10 = sub_16E8CB0(a1, a2, a3);
    v21 = 770;
    v19 = 1283;
    v18[0] = "Cannot find option named '";
    v18[1] = &s2;
    v20[0] = v18;
    v20[1] = "'!";
    result = sub_16B1F90(v6, v20, 0, 0, v10);
    if ( (_BYTE)result )
      return result;
    v12 = 0;
  }
  *((_DWORD *)v6 + 40) = v12;
  *((_DWORD *)v6 + 4) = v14;
  return 0;
}

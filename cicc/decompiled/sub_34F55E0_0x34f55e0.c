// Function: sub_34F55E0
// Address: 0x34f55e0
//
__int64 __fastcall sub_34F55E0(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // rbx
  int v5; // eax
  int v6; // r12d
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 result; // rax
  int v9; // eax
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 (__fastcall *v13)(__int64); // rax
  _DWORD *v14; // rcx
  unsigned int v15; // edx
  _QWORD v16[2]; // [rsp+0h] [rbp-50h] BYREF
  char v17; // [rsp+10h] [rbp-40h]

  v4 = a1;
  v5 = *(_DWORD *)(a1 + 44);
  if ( (v5 & 0xC) != 0 )
  {
    if ( (v5 & 8) != 0 )
    {
      v6 = 0;
      if ( *(_WORD *)(a1 + 68) != 20 )
        goto LABEL_4;
      while ( 1 )
      {
        v11 = *(_QWORD *)(v4 + 32);
        v12 = v11 + 40;
        v9 = *(_DWORD *)(v11 + 8);
        v10 = *(_DWORD *)(v12 + 8);
        if ( v9 == a2 )
          break;
        while ( 1 )
        {
          if ( a2 == v10 )
          {
            if ( v6 )
            {
              if ( v9 != v6 )
                return 0;
            }
            else
            {
              v6 = v9;
            }
          }
LABEL_10:
          v4 = *(_QWORD *)(v4 + 8);
          if ( (*(_BYTE *)(v4 + 44) & 8) == 0 )
            return 0;
          if ( *(_WORD *)(v4 + 68) == 20 )
            break;
LABEL_4:
          v7 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a3 + 520LL);
          if ( v7 == sub_2DCA430 )
            return 0;
          ((void (__fastcall *)(_QWORD *, __int64, __int64))v7)(v16, a3, v4);
          if ( !v17 )
            return 0;
          v9 = *(_DWORD *)(v16[0] + 8LL);
          v10 = *(_DWORD *)(v16[1] + 8LL);
          if ( v9 == a2 )
            goto LABEL_13;
        }
      }
LABEL_13:
      if ( !v6 )
      {
        v6 = v10;
        goto LABEL_10;
      }
      if ( v10 == v6 )
        goto LABEL_10;
    }
    return 0;
  }
  if ( *(_WORD *)(a1 + 68) != 20 )
  {
    v13 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a3 + 520LL);
    if ( v13 == sub_2DCA430 )
      return 0;
    ((void (__fastcall *)(_QWORD *, __int64, __int64))v13)(v16, a3, a1);
    if ( !v17 )
      return 0;
  }
  v14 = *(_DWORD **)(a1 + 32);
  if ( ((*v14 >> 8) & 0xFFF) != ((v14[10] >> 8) & 0xFFF) )
    return 0;
  v15 = v14[2];
  result = (unsigned int)v14[12];
  if ( a2 != v15 )
  {
    if ( a2 == (_DWORD)result )
      return v15;
    return 0;
  }
  return result;
}

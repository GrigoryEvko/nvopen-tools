// Function: sub_28C1CF0
// Address: 0x28c1cf0
//
_BOOL8 __fastcall sub_28C1CF0(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 a4, _QWORD *a5, _QWORD **a6, _QWORD *a7)
{
  _BOOL8 result; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdi
  unsigned int v11; // eax
  _QWORD *v12; // rdx
  __int64 v13; // rdx
  _BYTE *v14; // rax
  __int64 v15; // rdx
  _BYTE *v16; // rax
  unsigned int v19; // edx
  _QWORD *v20; // [rsp-18h] [rbp-18h]
  _QWORD **v21; // [rsp-10h] [rbp-10h]
  _QWORD **v22; // [rsp-10h] [rbp-10h]
  __int64 v23; // [rsp-10h] [rbp-10h]

  result = 0;
  if ( *a2 != 42 )
    return result;
  if ( *a3 != 54 )
    return 0;
  v8 = *((_QWORD *)a3 - 8);
  if ( !v8 )
    return 0;
  *a5 = v8;
  v9 = *((_QWORD *)a3 - 4);
  if ( *(_BYTE *)v9 == 17 )
  {
    *a6 = (_QWORD *)(v9 + 24);
  }
  else
  {
    v20 = a6;
    v23 = a4;
    v15 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 + 8) + 8LL) - 17;
    if ( (unsigned int)v15 > 1 )
      return 0;
    if ( *(_BYTE *)v9 > 0x15u )
      return 0;
    v16 = sub_AD7630(v9, 0, v15);
    if ( !v16 || *v16 != 17 )
      return 0;
    a6 = (_QWORD **)v20;
    a4 = v23;
    *v20 = v16 + 24;
  }
  if ( *(_BYTE *)a4 != 17 )
  {
    v22 = a6;
    v13 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a4 + 8) + 8LL) - 17;
    if ( (unsigned int)v13 <= 1 && *(_BYTE *)a4 <= 0x15u )
    {
      v14 = sub_AD7630(a4, 0, v13);
      if ( v14 )
      {
        if ( *v14 == 17 )
        {
          v10 = (__int64)(v14 + 24);
          a6 = v22;
          *a7 = v14 + 24;
          goto LABEL_9;
        }
      }
    }
    return 0;
  }
  v10 = a4 + 24;
  *a7 = a4 + 24;
LABEL_9:
  v11 = *(_DWORD *)(v10 + 8);
  if ( v11 <= 0x40 )
  {
    _RDX = *(_QWORD *)v10;
    __asm { tzcnt   rcx, rdx }
    v19 = 64;
    if ( *(_QWORD *)v10 )
      v19 = _RCX;
    if ( v11 > v19 )
      v11 = v19;
  }
  else
  {
    v21 = a6;
    v11 = sub_C44590(v10);
    a6 = v21;
  }
  v12 = (_QWORD *)**a6;
  if ( *((_DWORD *)*a6 + 2) > 0x40u )
    v12 = (_QWORD *)*v12;
  return v11 >= (unsigned __int64)v12;
}

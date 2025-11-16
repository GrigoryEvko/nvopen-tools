// Function: sub_28ED0D0
// Address: 0x28ed0d0
//
_BYTE *__fastcall sub_28ED0D0(unsigned __int8 *a1, __int64 a2)
{
  _BYTE *result; // rax
  unsigned __int8 *v3; // r12
  unsigned __int8 *v4; // rdx
  __int64 *v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // rax
  unsigned __int8 *v14; // rdx
  __int64 v15; // rdi
  _BYTE *v16; // r14
  _BYTE *v17; // rdi
  __int64 v18; // rdx
  _BYTE *v19; // rax
  _BYTE *v20; // rax

  result = (_BYTE *)*((_QWORD *)a1 + 2);
  if ( result )
  {
    v3 = a1;
    while ( 1 )
    {
      if ( *((_QWORD *)result + 1) )
        return result;
      result = (_BYTE *)*v3;
      if ( (unsigned __int8)result <= 0x1Cu )
        return result;
      if ( (_DWORD)result == 47 )
        break;
      if ( (_DWORD)result != 50 )
        return result;
      v4 = (v3[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v3 - 1) : &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
      if ( **(_BYTE **)v4 <= 0x15u )
      {
        result = (_BYTE *)*((_QWORD *)v4 + 4);
        if ( *result <= 0x15u )
          return result;
      }
      v5 = (__int64 *)sub_986520((__int64)v3);
      v6 = *v5;
      v7 = *(unsigned __int8 *)*v5;
      v8 = *v5 + 24;
      if ( (_BYTE)v7 == 18 )
        goto LABEL_13;
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17 <= 1 && (unsigned __int8)v7 <= 0x15u )
      {
        v20 = sub_AD7630(v6, 0, v7);
        if ( v20 )
        {
          v8 = (__int64)(v20 + 24);
          if ( *v20 == 18 )
          {
LABEL_13:
            v11 = v8;
            if ( *(void **)v8 == sub_C33340() )
              v11 = *(_QWORD *)(v8 + 8);
            if ( (*(_BYTE *)(v11 + 20) & 8) != 0 )
              goto LABEL_16;
          }
        }
LABEL_24:
        v5 = (__int64 *)sub_986520((__int64)v3);
      }
      v15 = v5[4];
      v16 = (_BYTE *)(v15 + 24);
      if ( *(_BYTE *)v15 == 18 )
        goto LABEL_26;
      v18 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v15 + 8) + 8LL) - 17;
      if ( (unsigned int)v18 <= 1 && *(_BYTE *)v15 <= 0x15u )
      {
        v19 = sub_AD7630(v15, 0, v18);
        if ( v19 )
        {
          if ( *v19 == 18 )
          {
            v16 = v19 + 24;
LABEL_26:
            v17 = v16;
            if ( *(void **)v16 == sub_C33340() )
              v17 = (_BYTE *)*((_QWORD *)v16 + 1);
            if ( (v17[20] & 8) == 0 )
              goto LABEL_19;
LABEL_16:
            v12 = *(unsigned int *)(a2 + 8);
            if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
            {
              sub_C8D5F0(a2, (const void *)(a2 + 16), v12 + 1, 8u, v9, v10);
              v12 = *(unsigned int *)(a2 + 8);
            }
            *(_QWORD *)(*(_QWORD *)a2 + 8 * v12) = v3;
            ++*(_DWORD *)(a2 + 8);
          }
        }
      }
LABEL_19:
      v13 = (_QWORD *)sub_986520((__int64)v3);
      sub_28ED0D0(*v13, a2);
      v3 = *(unsigned __int8 **)(sub_986520((__int64)v3) + 32);
      result = (_BYTE *)*((_QWORD *)v3 + 2);
      if ( !result )
        return result;
    }
    if ( (v3[7] & 0x40) != 0 )
      v14 = (unsigned __int8 *)*((_QWORD *)v3 - 1);
    else
      v14 = &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
    result = *(_BYTE **)v14;
    if ( **(_BYTE **)v14 <= 0x15u )
      return result;
    goto LABEL_24;
  }
  return result;
}

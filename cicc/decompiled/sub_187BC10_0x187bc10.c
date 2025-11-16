// Function: sub_187BC10
// Address: 0x187bc10
//
__int64 __fastcall sub_187BC10(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  unsigned int v4; // r14d
  unsigned __int8 v8; // dl
  _BYTE *v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rcx
  _QWORD *v13; // rdx
  int v15; // eax
  __int64 *v16; // r15
  unsigned __int64 v17; // rcx
  char *v18; // rcx
  _QWORD *v19; // r15
  __int64 v20; // rdx
  __int64 v21; // r15
  _BYTE *v22; // [rsp+0h] [rbp-50h] BYREF
  __int64 v23; // [rsp+8h] [rbp-48h]
  _BYTE v24[64]; // [rsp+10h] [rbp-40h] BYREF

  while ( 2 )
  {
    v8 = *(_BYTE *)(a3 + 16);
    LOBYTE(v4) = v8 == 3 || v8 == 0;
    if ( (_BYTE)v4 )
    {
LABEL_2:
      v22 = v24;
      v23 = 0x200000000LL;
      sub_1626560(a3, 19, (__int64)&v22);
      v9 = &v22[8 * (unsigned int)v23];
      if ( v22 == v9 )
      {
LABEL_23:
        v4 = 0;
      }
      else
      {
        v10 = (unsigned __int64)v22;
        while ( 1 )
        {
          v11 = *(unsigned int *)(*(_QWORD *)v10 + 8LL);
          if ( a1 == *(_QWORD *)(*(_QWORD *)v10 + 8 * (1 - v11)) )
          {
            v12 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v10 - 8 * v11) + 136LL);
            v13 = *(_QWORD **)(v12 + 24);
            if ( *(_DWORD *)(v12 + 32) > 0x40u )
              v13 = (_QWORD *)*v13;
            if ( a4 == v13 )
              break;
          }
          v10 += 8LL;
          if ( v9 == (_BYTE *)v10 )
            goto LABEL_23;
        }
        v4 = 1;
      }
      if ( v22 != v24 )
        _libc_free((unsigned __int64)v22);
    }
    else
    {
      while ( 1 )
      {
        if ( v8 == 56 )
        {
LABEL_24:
          LODWORD(v23) = 8 * sub_15A9520(a2, 0);
          if ( (unsigned int)v23 <= 0x40 )
            v22 = 0;
          else
            sub_16A4EF0((__int64)&v22, 0, 0);
          v4 = sub_1634900(a3, a2, (__int64)&v22);
          if ( (_BYTE)v4 )
          {
            v17 = (unsigned __int64)v22;
            if ( (unsigned int)v23 > 0x40 )
              v17 = *(_QWORD *)v22;
            v18 = (char *)a4 + v17;
            if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
              v19 = *(_QWORD **)(a3 - 8);
            else
              v19 = (_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
            v4 = sub_187BC10(a1, a2, *v19, v18);
          }
          if ( (unsigned int)v23 > 0x40 && v22 )
            j_j___libc_free_0_0(v22);
          return v4;
        }
        if ( v8 == 5 )
        {
          v15 = *(unsigned __int16 *)(a3 + 18);
          if ( (_WORD)v15 == 32 )
            goto LABEL_24;
        }
        else
        {
          if ( v8 <= 0x17u )
            return v4;
          v15 = v8 - 24;
        }
        if ( v15 != 47 )
          break;
        if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
          v16 = *(__int64 **)(a3 - 8);
        else
          v16 = (__int64 *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
        a3 = *v16;
        v8 = *(_BYTE *)(a3 + 16);
        if ( !v8 || v8 == 3 )
          goto LABEL_2;
      }
      if ( v15 == 55 )
      {
        v20 = (*(_BYTE *)(a3 + 23) & 0x40) != 0 ? *(_QWORD *)(a3 - 8) : a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
        if ( (unsigned __int8)sub_187BC10(a1, a2, *(_QWORD *)(v20 + 24), a4) )
        {
          if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
            v21 = *(_QWORD *)(a3 - 8);
          else
            v21 = a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
          a3 = *(_QWORD *)(v21 + 48);
          continue;
        }
      }
    }
    return v4;
  }
}

// Function: sub_1F4BB70
// Address: 0x1f4bb70
//
__int64 __fastcall sub_1F4BB70(_QWORD *a1, __int64 a2, unsigned int a3, __int64 a4, unsigned int a5)
{
  unsigned int v10; // r12d
  _WORD *v12; // rsi
  __int64 v13; // rdx
  _BYTE *v14; // rax
  _BYTE *v15; // rcx
  __int16 *v16; // rax
  int v17; // r8d
  int v18; // r13d
  _WORD *v19; // rax
  __int64 v20; // rdi
  _WORD *v21; // rsi
  __int64 v22; // rdx
  _BYTE *v23; // rax
  _BYTE *v24; // rcx
  unsigned int *v25; // rax
  unsigned int *v26; // rcx
  unsigned int v27; // edx
  int v28; // eax
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r12
  __int16 v33; // ax
  unsigned int v34; // [rsp+Ch] [rbp-34h]

  if ( sub_1F4B670((__int64)a1) || sub_1F4B690((__int64)a1) )
  {
    if ( !sub_1F4B690((__int64)a1) )
    {
      v12 = sub_1F4B8B0((__int64)a1, a2);
      if ( a3 )
      {
        v13 = a3 - 1;
        v14 = *(_BYTE **)(a2 + 32);
        a3 = 0;
        v15 = &v14[40 * v13];
        while ( 1 )
        {
          if ( !*v14 )
            a3 -= ((v14[3] & 0x10) == 0) - 1;
          if ( v14 == v15 )
            break;
          v14 += 40;
        }
      }
      if ( (unsigned __int16)v12[4] <= a3 )
      {
        v33 = **(_WORD **)(a2 + 16);
        switch ( v33 )
        {
          case 0:
          case 8:
          case 10:
          case 14:
          case 15:
          case 45:
            return 0;
          default:
            switch ( v33 )
            {
              case 2:
              case 3:
              case 4:
              case 6:
              case 9:
              case 12:
              case 13:
              case 17:
              case 18:
                return 0;
              default:
                return sub_1F3BC50(a1[23], (__int64)a1, a2);
            }
        }
      }
      else
      {
        v16 = (__int16 *)(*(_QWORD *)(a1[22] + 144LL) + 4LL * (a3 + (unsigned __int16)v12[3]));
        v17 = *v16;
        if ( v17 < 0 )
          v17 = 1000;
        v34 = v17;
        v10 = v17;
        if ( !a4 )
          return v10;
        v18 = (unsigned __int16)v16[1];
        v19 = sub_1F4B8B0((__int64)a1, a4);
        v20 = (unsigned __int16)v19[6];
        v21 = v19;
        if ( !(_WORD)v20 )
          return v10;
        if ( a5 )
        {
          v22 = a5 - 1;
          v23 = *(_BYTE **)(a4 + 32);
          a5 = 0;
          v24 = &v23[40 * v22];
          while ( 1 )
          {
            if ( !*v23 && (v23[4] & 1) == 0 && (v23[4] & 2) == 0 )
              a5 += (v23[3] & 0x10) == 0;
            if ( v23 == v24 )
              break;
            v23 += 40;
          }
        }
        v25 = (unsigned int *)(*(_QWORD *)(a1[22] + 152LL) + 12LL * (unsigned __int16)v21[5]);
        v26 = &v25[3 * v20];
        while ( 1 )
        {
          if ( *v25 >= a5 )
          {
            if ( *v25 > a5 )
              return v10;
            v27 = v25[1];
            if ( !v27 || v18 == v27 )
              break;
          }
          v25 += 3;
          if ( v26 == v25 )
            return v10;
        }
        v28 = v25[2];
        if ( v28 <= 0 || v28 <= v34 )
          return v34 - v28;
      }
      return 0;
    }
    if ( a4 )
    {
      v10 = (*(__int64 (__fastcall **)(_QWORD, _QWORD *, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)a1[23] + 840LL))(
              a1[23],
              a1 + 9,
              a2,
              a3,
              a4,
              a5);
    }
    else
    {
      v30 = a1[21];
      if ( !v30 )
        goto LABEL_37;
      v31 = v30 + 10LL * *(unsigned __int16 *)(*(_QWORD *)(a2 + 16) + 6LL);
      v32 = *(unsigned __int16 *)(v31 + 6) + a3;
      if ( *(unsigned __int16 *)(v31 + 8) <= (unsigned int)v32 )
        goto LABEL_37;
      v10 = *(_DWORD *)(a1[19] + 4 * v32);
    }
    if ( (v10 & 0x80000000) == 0 )
      return v10;
LABEL_37:
    v10 = (*(__int64 (__fastcall **)(_QWORD, _QWORD *, __int64, _QWORD))(*(_QWORD *)a1[23] + 848LL))(
            a1[23],
            a1 + 9,
            a2,
            0);
    v29 = sub_1F3BC50(a1[23], (__int64)a1, a2);
    if ( v10 < v29 )
      return v29;
    return v10;
  }
  return sub_1F3BC50(a1[23], (__int64)a1, a2);
}

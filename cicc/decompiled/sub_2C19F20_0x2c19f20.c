// Function: sub_2C19F20
// Address: 0x2c19f20
//
__int64 __fastcall sub_2C19F20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  _BYTE *v5; // rsi
  __int64 v6; // r13
  __int64 v8; // rax
  int v9; // edx
  int v10; // r12d
  _QWORD *v11; // r14
  _QWORD *v12; // rbx
  unsigned __int64 v13; // rsi
  _QWORD *v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // rdi
  int v20; // esi
  __int64 v21; // rcx
  __int64 v22; // rdx

  v3 = *(unsigned __int8 *)(a1 + 8);
  switch ( (char)v3 )
  {
    case 0:
    case 3:
    case 5:
    case 19:
    case 20:
    case 21:
    case 22:
    case 26:
      if ( (_BYTE)v3 == 5 )
      {
        v5 = *(_BYTE **)(*(_QWORD *)(a1 + 96) + 48LL);
      }
      else
      {
        v3 = (unsigned int)(v3 - 19);
        if ( (unsigned __int8)v3 > 3u )
          return (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 32LL))(a1, a2, a3);
        v5 = *(_BYTE **)(a1 + 96);
      }
      if ( !v5 )
        return (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 32LL))(a1, a2, a3);
LABEL_4:
      if ( BYTE4(a2) )
      {
        if ( (unsigned __int8)sub_2AB6B40(a3, (__int64)v5, (_DWORD)a2 != 0, v3) )
          return 0;
      }
      else if ( (unsigned __int8)sub_2AB6B40(a3, (__int64)v5, (unsigned int)a2 > 1, v3) )
      {
        return 0;
      }
      v8 = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 32LL))(a1, a2, a3);
      v10 = v9;
      v6 = v8;
      v11 = sub_C52410();
      v12 = v11 + 1;
      v13 = sub_C959E0();
      v14 = (_QWORD *)v11[2];
      if ( v14 )
      {
        v15 = v11 + 1;
        do
        {
          while ( 1 )
          {
            v16 = v14[2];
            v17 = v14[3];
            if ( v13 <= v14[4] )
              break;
            v14 = (_QWORD *)v14[3];
            if ( !v17 )
              goto LABEL_14;
          }
          v15 = v14;
          v14 = (_QWORD *)v14[2];
        }
        while ( v16 );
LABEL_14:
        if ( v12 != v15 && v13 >= v15[4] )
          v12 = v15;
      }
      if ( v12 != (_QWORD *)((char *)sub_C52410() + 8) )
      {
        v18 = v12[7];
        if ( v18 )
        {
          v19 = v12 + 6;
          v20 = qword_500DDC0[1];
          do
          {
            while ( 1 )
            {
              v21 = *(_QWORD *)(v18 + 16);
              v22 = *(_QWORD *)(v18 + 24);
              if ( *(_DWORD *)(v18 + 32) >= v20 )
                break;
              v18 = *(_QWORD *)(v18 + 24);
              if ( !v22 )
                goto LABEL_23;
            }
            v19 = (_QWORD *)v18;
            v18 = *(_QWORD *)(v18 + 16);
          }
          while ( v21 );
LABEL_23:
          if ( v19 != v12 + 6 && v20 >= *((_DWORD *)v19 + 8) && *((int *)v19 + 9) > 0 && !v10 )
            return LODWORD(qword_500DDC0[17]);
        }
      }
      return v6;
    case 1:
    case 2:
    case 4:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 23:
    case 24:
    case 25:
    case 27:
    case 28:
    case 29:
    case 30:
    case 31:
    case 32:
    case 33:
    case 34:
    case 35:
    case 36:
      v5 = *(_BYTE **)(a1 + 136);
      if ( v5 && *v5 > 0x1Cu )
        goto LABEL_4;
      return (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 32LL))(a1, a2, a3);
    default:
      BUG();
  }
}

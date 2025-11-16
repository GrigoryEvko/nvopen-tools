// Function: sub_76C8B0
// Address: 0x76c8b0
//
__int64 __fastcall sub_76C8B0(__int64 a1, __int64 a2)
{
  __int64 (*v3)(void); // rax
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdi
  _QWORD *i; // rbx
  __int64 v12; // rdi
  _QWORD *v13; // rax
  _QWORD *v14; // rbx
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rbx
  __int64 v19; // rdi
  __int64 v20; // rdi
  _QWORD *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 v32; // rbx
  _QWORD *v33; // rbx
  __int64 v34; // rdi
  _QWORD *v35; // rbx

  v3 = *(__int64 (**)(void))(a2 + 48);
  if ( !v3 )
    goto LABEL_95;
  result = v3();
  if ( *(_DWORD *)(a2 + 72) )
    return result;
  if ( *(_DWORD *)(a2 + 76) )
  {
    *(_DWORD *)(a2 + 76) = 0;
LABEL_12:
    result = *(_QWORD *)(a2 + 56);
    if ( result )
      return ((__int64 (__fastcall *)(__int64, __int64))result)(a1, a2);
  }
  else
  {
LABEL_95:
    result = *(unsigned __int8 *)(a1 + 40);
    switch ( *(_BYTE *)(a1 + 40) )
    {
      case 0:
      case 0x17:
        goto LABEL_7;
      case 1:
      case 3:
      case 4:
        v6 = *(_QWORD *)(a1 + 48);
        if ( v6 )
          result = sub_76CDC0(v6);
        if ( *(_DWORD *)(a2 + 72) )
          return result;
        v7 = *(_QWORD *)(a1 + 72);
        if ( v7 )
        {
          result = sub_76C8B0(v7, a2);
          if ( *(_DWORD *)(a2 + 72) )
            return result;
        }
        v8 = *(_QWORD *)(a1 + 80);
        if ( v8 )
          goto LABEL_20;
        goto LABEL_8;
      case 2:
        result = sub_76CDC0(*(_QWORD *)(a1 + 48));
        if ( *(_DWORD *)(a2 + 72) )
          return result;
        v13 = *(_QWORD **)(a1 + 72);
        if ( !*v13 )
          goto LABEL_37;
        result = sub_76C8B0(*v13, a2);
        if ( *(_DWORD *)(a2 + 72) )
          return result;
        v13 = *(_QWORD **)(a1 + 72);
LABEL_37:
        v8 = v13[1];
        if ( v8 )
LABEL_20:
          sub_76C8B0(v8, a2);
        goto LABEL_8;
      case 5:
        result = sub_76CDC0(*(_QWORD *)(a1 + 48));
        if ( *(_DWORD *)(a2 + 72) )
          return result;
        v8 = *(_QWORD *)(a1 + 72);
        if ( !v8 )
          goto LABEL_12;
        goto LABEL_20;
      case 6:
      case 7:
      case 9:
      case 0x14:
      case 0x16:
      case 0x18:
        goto LABEL_8;
      case 8:
        v9 = *(_QWORD *)(a1 + 72);
        if ( !v9 )
          goto LABEL_22;
        goto LABEL_52;
      case 0xA:
LABEL_22:
        v10 = *(_QWORD *)(a1 + 48);
        if ( v10 )
          goto LABEL_23;
        goto LABEL_8;
      case 0xB:
        sub_76D840(*(_QWORD *)(a1 + 72), a2);
        if ( qword_4F04C50 )
        {
          if ( *(_QWORD *)(qword_4F04C50 + 80LL) == a1 )
          {
            for ( i = *(_QWORD **)(qword_4F04C50 + 216LL); i; i = (_QWORD *)*i )
              sub_76CDC0(i[1]);
          }
        }
        goto LABEL_8;
      case 0xC:
        v5 = *(_QWORD *)(a1 + 72);
        if ( v5 )
        {
          sub_76C8B0(v5, a2);
          result = *(unsigned int *)(a2 + 72);
          if ( (_DWORD)result )
            return result;
        }
LABEL_7:
        sub_76CDC0(*(_QWORD *)(a1 + 48));
        goto LABEL_8;
      case 0xD:
        v14 = *(_QWORD **)(a1 + 80);
        if ( *v14 )
        {
          sub_76C8B0(*v14, a2);
          result = *(unsigned int *)(a2 + 72);
          if ( (_DWORD)result )
            return result;
        }
        v15 = *(_QWORD *)(a1 + 48);
        if ( v15 )
        {
          sub_76CDC0(v15);
          result = *(unsigned int *)(a2 + 72);
          if ( (_DWORD)result )
            return result;
        }
        v16 = *(_QWORD *)(a1 + 72);
        if ( v16 )
        {
          sub_76C8B0(v16, a2);
          result = *(unsigned int *)(a2 + 72);
          if ( (_DWORD)result )
            return result;
        }
        v17 = v14[1];
        if ( !v17 )
          goto LABEL_8;
        sub_76CDC0(v17);
        result = *(unsigned int *)(a2 + 72);
        if ( (_DWORD)result )
          return result;
        goto LABEL_12;
      case 0xE:
        v21 = *(_QWORD **)(a1 + 80);
        v22 = v21[1];
        if ( v22 )
        {
          v23 = *(_QWORD *)(v22 + 184);
          if ( v23 )
            sub_76D400(v23, a2);
        }
        v24 = v21[2];
        if ( v24 )
        {
          v25 = *(_QWORD *)(v24 + 184);
          if ( v25 )
            sub_76D400(v25, a2);
        }
        v26 = v21[5];
        if ( v26 )
        {
          v27 = *(_QWORD *)(v26 + 184);
          if ( v27 )
            sub_76D400(v27, a2);
        }
        v28 = v21[6];
        if ( v28 )
        {
          v29 = *(_QWORD *)(v28 + 184);
          if ( v29 )
            sub_76D400(v29, a2);
        }
        v30 = v21[7];
        if ( v30 )
          sub_76CDC0(v30);
        v31 = v21[8];
        if ( v31 )
          sub_76CDC0(v31);
        v12 = *(_QWORD *)(a1 + 72);
        if ( !v12 )
          goto LABEL_8;
        goto LABEL_31;
      case 0xF:
        if ( !*(_DWORD *)(a2 + 84) )
          goto LABEL_8;
        v18 = *(_QWORD *)(a1 + 80);
        v19 = *(_QWORD *)(v18 + 8);
        if ( !v19 )
          goto LABEL_8;
        result = sub_76D560(v19, a2);
        if ( *(_DWORD *)(a2 + 72) )
          return result;
        v20 = *(_QWORD *)(v18 + 16);
        if ( !v20 )
          goto LABEL_12;
        sub_76D560(v20, a2);
LABEL_8:
        result = *(_QWORD *)(a2 + 56);
        if ( !result || *(_DWORD *)(a2 + 72) )
          return result;
        return ((__int64 (__fastcall *)(__int64, __int64))result)(a1, a2);
      case 0x10:
        result = sub_76CDC0(*(_QWORD *)(a1 + 48));
        if ( *(_DWORD *)(a2 + 72) )
          return result;
        v12 = *(_QWORD *)(a1 + 72);
        if ( !v12 )
          goto LABEL_12;
LABEL_31:
        result = sub_76C8B0(v12, a2);
        if ( *(_DWORD *)(a2 + 72) )
          return result;
        goto LABEL_12;
      case 0x11:
        v9 = *(_QWORD *)(a1 + 72);
        goto LABEL_52;
      case 0x12:
        v35 = *(_QWORD **)(*(_QWORD *)(a1 + 72) + 136LL);
        if ( !v35 )
          goto LABEL_8;
        while ( 1 )
        {
          result = sub_76CDC0(v35[5]);
          if ( *(_DWORD *)(a2 + 72) )
            return result;
          v35 = (_QWORD *)*v35;
          if ( !v35 )
            goto LABEL_12;
        }
      case 0x13:
        v32 = *(_QWORD *)(a1 + 72);
        result = sub_76C8B0(*(_QWORD *)(v32 + 8), a2);
        if ( *(_DWORD *)(a2 + 72) )
          return result;
        v33 = *(_QWORD **)(v32 + 16);
        if ( !v33 )
          goto LABEL_12;
        break;
      case 0x15:
        v10 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL);
        goto LABEL_23;
      case 0x19:
        v9 = *(_QWORD *)(a1 + 72);
        if ( v9 )
        {
LABEL_52:
          sub_76D400(v9, a2);
        }
        else
        {
          v10 = *(_QWORD *)(a1 + 48);
          if ( !v10 )
LABEL_91:
            sub_721090();
LABEL_23:
          sub_76CDC0(v10);
        }
        goto LABEL_8;
      default:
        goto LABEL_91;
    }
    while ( 1 )
    {
      v34 = v33[4];
      if ( v34 )
      {
        result = sub_76D400(v34, a2);
        if ( *(_DWORD *)(a2 + 72) )
          break;
      }
      result = sub_76C8B0(v33[3], a2);
      if ( *(_DWORD *)(a2 + 72) )
        break;
      v33 = (_QWORD *)*v33;
      if ( !v33 )
        goto LABEL_12;
    }
  }
  return result;
}

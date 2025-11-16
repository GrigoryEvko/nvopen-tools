// Function: sub_7B0EB0
// Address: 0x7b0eb0
//
__int64 __fastcall sub_7B0EB0(unsigned __int64 a1, __int64 a2)
{
  int v3; // r13d
  _BYTE *v4; // r15
  __int16 v5; // bx
  __int64 *v6; // r14
  __int64 v7; // r9
  unsigned __int64 v8; // rdi
  unsigned int *v9; // r10
  int v10; // r11d
  unsigned __int64 *v11; // r8
  int v12; // esi
  int v13; // edx
  __int16 v14; // r13
  __int16 v15; // r11
  __int64 result; // rax
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 *v19; // r15
  int v20; // eax
  __int64 *v21; // rax
  __int64 v22; // [rsp+0h] [rbp-40h]
  unsigned __int64 *v23; // [rsp+0h] [rbp-40h]
  int v24; // [rsp+Ch] [rbp-34h]

  v3 = dword_4F17F78;
  if ( dword_4F17F78 )
  {
    result = qword_4F17F70;
    *(_QWORD *)a2 = qword_4F17F70;
    return result;
  }
  v4 = (_BYTE *)a1;
  v22 = unk_4F06458;
  v5 = qword_4F06498;
  v24 = unk_4F0647C;
  if ( qword_4F06498 <= a1 && unk_4F06490 > a1 )
    goto LABEL_4;
  v17 = sub_7AEFF0(a1);
  v6 = v17;
  if ( (v17[6] & 0x40) != 0 )
  {
    if ( v17[7] == a1 )
    {
      result = v17[11];
      *(_QWORD *)a2 = result;
      return result;
    }
    if ( a1 != v17[8] - 1 )
      sub_721090();
    v18 = v17[2];
    if ( !v18 )
    {
      *(_QWORD *)a2 = v6[11];
      result = (unsigned int)*(unsigned __int16 *)(a2 + 4) + *((_DWORD *)v6 + 16) - *((_DWORD *)v6 + 14) - 1;
      *(_WORD *)(a2 + 4) = *(_WORD *)(a2 + 4) + *((_WORD *)v6 + 32) - *((_WORD *)v6 + 28) - 1;
      return result;
    }
    v4 = (_BYTE *)(v18 + v6[4] - 1);
    if ( (unsigned __int64)v4 < qword_4F06498 || (unsigned __int64)v4 >= unk_4F06490 )
    {
      v19 = sub_7AEFF0((unsigned __int64)v4);
      v6 = v19;
      if ( *((_DWORD *)v19 + 22) )
      {
LABEL_38:
        result = v19[11];
        *(_QWORD *)a2 = result;
        goto LABEL_30;
      }
      while ( 1 )
      {
        if ( (v19[6] & 4) != 0 )
        {
          v21 = (__int64 *)v19[3];
          if ( !v21 )
            goto LABEL_50;
        }
        else
        {
          v21 = sub_7AF170((__int64)v19);
          if ( !v21 )
          {
LABEL_50:
            v4 = (_BYTE *)v19[2];
            if ( !v4 )
            {
              v4 = (_BYTE *)qword_4F06498;
              if ( unk_4F06478 )
                v4 = (_BYTE *)(unk_4F06470 + qword_4F06498);
            }
            goto LABEL_5;
          }
        }
        v19 = v21;
LABEL_47:
        if ( *((_DWORD *)v19 + 22) )
          goto LABEL_38;
      }
    }
LABEL_4:
    v6 = 0;
    goto LABEL_5;
  }
  if ( a1 < qword_4F06498 || a1 >= unk_4F06490 )
  {
    v19 = sub_7AEFF0(a1);
    goto LABEL_47;
  }
LABEL_5:
  v7 = qword_4F084E0;
  if ( !qword_4F084E0 || (v8 = *(_QWORD *)(qword_4F084E0 + 8), v8 > (unsigned __int64)v4) )
  {
    v7 = v22;
    if ( !v22 )
    {
      v9 = (unsigned int *)&unk_4F06480;
      v14 = 1;
      goto LABEL_28;
    }
    v8 = *(_QWORD *)(v22 + 8);
  }
  v9 = (unsigned int *)&unk_4F06480;
  v10 = unk_4F06480;
  v11 = &qword_4F06488[unk_4F06480 - 1];
  while ( (unsigned __int64)v4 >= v8 )
  {
    v13 = *(_DWORD *)(v7 + 16);
    if ( (unsigned int)(v13 - 1) > 1 )
    {
      if ( v4 != (_BYTE *)v8 )
      {
        v12 = v3 - 1;
        if ( v13 != 3 )
          v12 = v3 + 2;
        v3 = v12;
      }
LABEL_12:
      v7 = *(_QWORD *)v7;
      if ( !v7 )
        break;
      goto LABEL_13;
    }
    if ( !*v4 && v4[1] == 2 && v4 == (_BYTE *)v8 )
      break;
    v5 = v8 + 2;
    if ( v13 != 2 )
      v5 = v8;
    v3 = 0;
    v24 = *(_DWORD *)(v7 + 24);
    if ( v10 )
    {
      v3 = v10;
      if ( *v11 > v8 )
      {
        v23 = v11;
        v20 = sub_7AB680(v8);
        v11 = v23;
        v3 = v20;
      }
    }
    if ( qword_4F084E0 && *(_QWORD *)(qword_4F084E0 + 8) >= v8 )
      goto LABEL_12;
    qword_4F084E0 = v7;
    v7 = *(_QWORD *)v7;
    if ( !v7 )
      break;
LABEL_13:
    v8 = *(_QWORD *)(v7 + 8);
  }
  v14 = v3 + 1;
LABEL_28:
  v15 = (_WORD)v4 - v5;
  *(_DWORD *)a2 = v24;
  result = *v9;
  if ( (_DWORD)result && qword_4F06488[(int)result - 1] > (unsigned __int64)v4 )
    result = sub_7AB680((unsigned __int64)v4);
  *(_WORD *)(a2 + 4) = v14 + v15 - result;
LABEL_30:
  if ( v6 )
  {
    v6[11] = *(_QWORD *)a2;
    qword_4F061D0 = *(_QWORD *)a2;
    return (__int64)&qword_4F061D0;
  }
  return result;
}

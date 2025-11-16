// Function: sub_8CC670
// Address: 0x8cc670
//
__int64 *__fastcall sub_8CC670(unsigned __int8 *a1)
{
  __int64 *result; // rax
  _QWORD *v2; // rdx
  char v3; // dl
  _QWORD *v4; // rcx
  __int64 v5; // rcx

  result = (__int64 *)a1[140];
  switch ( a1[140] )
  {
    case 1u:
      v2 = (_QWORD *)qword_4F602A0;
      if ( qword_4F602A0 )
        return sub_8CBB20(6u, (__int64)a1, v2);
      qword_4F602A0 = (__int64)a1;
      if ( (*(a1 - 8) & 2) == 0 )
        return result;
      return sub_8C7090(6, (__int64)a1);
    case 2u:
      result = (__int64 *)a1[162];
      if ( ((unsigned __int8)result & 4) != 0 )
      {
        v2 = (_QWORD *)qword_4F60270;
        if ( qword_4F60270 )
          return sub_8CBB20(6u, (__int64)a1, v2);
        qword_4F60270 = (__int64)a1;
        if ( (*(a1 - 8) & 2) != 0 )
          return sub_8C7090(6, (__int64)a1);
        return result;
      }
      v3 = a1[161];
      if ( (v3 & 0x40) != 0 )
      {
        v2 = (_QWORD *)qword_4F60290;
        if ( qword_4F60290 )
          return sub_8CBB20(6u, (__int64)a1, v2);
        qword_4F60290 = (__int64)a1;
        if ( (*(a1 - 8) & 2) != 0 )
          return sub_8C7090(6, (__int64)a1);
        return result;
      }
      if ( v3 < 0 )
      {
        v2 = (_QWORD *)qword_4F60288;
        if ( qword_4F60288 )
          return sub_8CBB20(6u, (__int64)a1, v2);
        qword_4F60288 = (__int64)a1;
        if ( (*(a1 - 8) & 2) != 0 )
          return sub_8C7090(6, (__int64)a1);
        return result;
      }
      if ( ((unsigned __int8)result & 1) != 0 )
      {
        v2 = (_QWORD *)qword_4F60280;
        if ( qword_4F60280 )
          return sub_8CBB20(6u, (__int64)a1, v2);
        qword_4F60280 = (__int64)a1;
        if ( (*(a1 - 8) & 2) != 0 )
          return sub_8C7090(6, (__int64)a1);
        return result;
      }
      if ( ((unsigned __int8)result & 2) == 0 )
      {
        result = (__int64 *)a1[160];
        if ( (a1[161] & 1) != 0 )
        {
          v4 = qword_4F60440;
          v2 = (_QWORD *)qword_4F60440[(_QWORD)result];
          if ( v2 )
            return sub_8CBB20(6u, (__int64)a1, v2);
        }
        else
        {
          v4 = qword_4F604C0;
          v2 = (_QWORD *)qword_4F604C0[(_QWORD)result];
          if ( v2 )
            return sub_8CBB20(6u, (__int64)a1, v2);
        }
        v4[(_QWORD)result] = a1;
        if ( (*(a1 - 8) & 2) != 0 )
          return sub_8C7090(6, (__int64)a1);
        return result;
      }
      v2 = (_QWORD *)qword_4F60278;
      if ( qword_4F60278 )
        return sub_8CBB20(6u, (__int64)a1, v2);
      qword_4F60278 = (__int64)a1;
      if ( (*(a1 - 8) & 2) == 0 )
        return result;
      return sub_8C7090(6, (__int64)a1);
    case 3u:
      v5 = a1[160];
      result = qword_4F603C0;
      v2 = (_QWORD *)qword_4F603C0[v5];
      if ( !v2 )
        goto LABEL_14;
      return sub_8CBB20(6u, (__int64)a1, v2);
    case 4u:
      v5 = a1[160];
      result = qword_4F602C0;
      v2 = (_QWORD *)qword_4F602C0[v5];
      if ( !v2 )
        goto LABEL_14;
      return sub_8CBB20(6u, (__int64)a1, v2);
    case 5u:
      v5 = a1[160];
      result = qword_4F60340;
      v2 = (_QWORD *)qword_4F60340[v5];
      if ( v2 )
        return sub_8CBB20(6u, (__int64)a1, v2);
LABEL_14:
      result[v5] = (__int64)a1;
      if ( (*(a1 - 8) & 2) != 0 )
        return sub_8C7090(6, (__int64)a1);
      return result;
    case 0x13u:
      if ( (a1[141] & 0x20) != 0 )
      {
        v2 = (_QWORD *)qword_4F60260;
        if ( !qword_4F60260 )
        {
          qword_4F60260 = (__int64)a1;
          if ( (*(a1 - 8) & 2) == 0 )
            return result;
          return sub_8C7090(6, (__int64)a1);
        }
        return sub_8CBB20(6u, (__int64)a1, v2);
      }
      v2 = (_QWORD *)qword_4F60268;
      if ( qword_4F60268 )
        return sub_8CBB20(6u, (__int64)a1, v2);
      qword_4F60268 = (__int64)a1;
      if ( (*(a1 - 8) & 2) != 0 )
        return sub_8C7090(6, (__int64)a1);
      return result;
    case 0x14u:
      v2 = (_QWORD *)qword_4F60298;
      if ( qword_4F60298 )
        return sub_8CBB20(6u, (__int64)a1, v2);
      qword_4F60298 = (__int64)a1;
      if ( (*(a1 - 8) & 2) == 0 )
        return result;
      return sub_8C7090(6, (__int64)a1);
    default:
      sub_721090();
  }
}

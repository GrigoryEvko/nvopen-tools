// Function: sub_89A800
// Address: 0x89a800
//
_BYTE *__fastcall sub_89A800(__int64 a1)
{
  _BYTE *v1; // rbx
  _BYTE *v2; // r12
  int v4; // r15d
  _QWORD *v5; // r10
  __int64 v6; // rsi
  char v7; // al
  __int64 v8; // rax
  __int64 *v9; // rsi
  __int64 *v10; // rax
  __int64 *v11; // rdi
  _DWORD *v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // rax
  __int64 *v15; // [rsp+0h] [rbp-60h]
  _QWORD *v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+10h] [rbp-50h]
  _QWORD *v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 *v20; // [rsp+20h] [rbp-40h] BYREF
  _QWORD *v21[7]; // [rsp+28h] [rbp-38h] BYREF

  v1 = (_BYTE *)a1;
  if ( dword_4F077C4 != 2 )
    return v1;
  if ( (_DWORD)qword_4F077B4 )
  {
    if ( qword_4F077A0 )
      return v1;
  }
  else if ( HIDWORD(qword_4F077B4) && qword_4F077A8 <= 0x9FC3u )
  {
    return v1;
  }
  if ( *(_BYTE *)(a1 + 120) == 1 )
  {
    v19 = *(_QWORD *)(a1 + 192);
    if ( v19 )
    {
      v17 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)a1 + 88LL) + 32LL);
      v4 = *(_DWORD *)(sub_892BC0(v17) + 4);
      if ( *(_BYTE *)(a1 + 120) == 1 )
      {
        v5 = (_QWORD *)v17;
        while ( *(_BYTE *)(v19 + 140) == 12 && *(_BYTE *)(v19 + 184) == 10 )
        {
          v6 = *(_QWORD *)(v19 + 160);
          v7 = *(_BYTE *)(v6 + 140);
          v19 = v6;
          if ( (unsigned __int8)(v7 - 9) > 2u )
          {
            if ( v7 != 12 || *(_BYTE *)(v6 + 184) != 10 )
              return v1;
            v14 = *(_QWORD *)(v6 + 168);
            v2 = *(_BYTE **)(v14 + 16);
            v9 = *(__int64 **)v14;
          }
          else
          {
            v8 = *(_QWORD *)(v6 + 168);
            v2 = *(_BYTE **)(v8 + 160);
            if ( !v2 || v2[120] != 1 )
              return v1;
            v9 = *(__int64 **)(v8 + 168);
          }
          v16 = v5;
          v10 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)v2 + 88LL) + 32LL);
          v15 = v10;
          if ( !v10 || *(_DWORD *)(sub_892BC0(*v10) + 4) != 1 )
            return v1;
          sub_89A1A0(v16, v9, v21, &v20);
          v11 = v20;
          if ( v20 )
          {
            while ( v21[0] )
            {
              v12 = (_DWORD *)sub_892B20((__int64)v11);
              if ( !v12 || v4 != v12[1] )
              {
                if ( v20 )
                  return v1;
                goto LABEL_35;
              }
              if ( *v12 != *((_DWORD *)v21[0] + 15) )
                return v1;
              sub_89A1C0((__int64 *)v21, &v20);
              v11 = v20;
              if ( !v20 )
                goto LABEL_35;
            }
            return v1;
          }
LABEL_35:
          if ( v21[0] )
            return v1;
          v13 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)v1 + 88LL) + 32LL);
          if ( *(_QWORD *)(v13[4] + 16LL) )
            return v1;
          v18 = (_QWORD *)*v13;
          if ( !(unsigned int)sub_89B3C0(*v13, *v15, 0, 10, 0, 8) )
            return v1;
          if ( v2[120] != 1 )
            return v2;
          v5 = v18;
          v1 = v2;
        }
      }
    }
  }
  return v1;
}

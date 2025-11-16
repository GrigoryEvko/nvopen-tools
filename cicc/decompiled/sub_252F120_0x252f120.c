// Function: sub_252F120
// Address: 0x252f120
//
__int64 __fastcall sub_252F120(__int64 a1, __int64 a2, char a3)
{
  unsigned int v3; // r12d
  unsigned int v4; // r14d
  _BYTE **v6; // r12
  unsigned __int8 *v8; // rdi
  int v9; // ecx
  unsigned __int8 *v10; // rdi
  __int64 v11; // rdi
  bool v12; // al
  _BYTE *v13; // rbx
  __int64 v14; // rdi
  char v15; // [rsp+Ch] [rbp-34h]
  __int64 v16; // [rsp+10h] [rbp-30h] BYREF
  __int64 v17[5]; // [rsp+18h] [rbp-28h] BYREF

  v3 = (*(_DWORD *)(a2 + 96) >> 3) & 1;
  LOBYTE(v3) = *(_DWORD *)(a2 + 96) == 17 || (*(_DWORD *)(a2 + 96) & 8) != 0;
  if ( (_BYTE)v3 )
  {
    v4 = *(unsigned __int8 *)(a2 + 24);
    if ( !(_BYTE)v4 )
      return v3;
    v6 = *(_BYTE ***)a1;
    v8 = *(unsigned __int8 **)(a2 + 16);
    if ( v8 )
    {
      v9 = *v8;
      if ( (unsigned int)(v9 - 12) <= 1 )
      {
LABEL_6:
        if ( **(_BYTE **)(a1 + 8) == 1 && !a3 )
        {
          v10 = *(unsigned __int8 **)(a2 + 16);
          v3 = **(unsigned __int8 **)(a1 + 16);
          if ( !(_BYTE)v3 )
          {
            if ( !v10 || (unsigned int)*v10 - 12 > 1 || **(_BYTE **)(a1 + 24) )
              return v3;
LABEL_12:
            v3 = 0;
            v16 = sub_250C3F0((unsigned __int64)v10, *(_QWORD *)(**(_QWORD **)(a1 + 32) + 8LL));
            if ( v16 )
            {
              v3 = v4;
              sub_252E900(*(_QWORD *)(a1 + 40), &v16);
              if ( **(_QWORD **)(a1 + 48) )
              {
                v11 = *(_QWORD *)(a1 + 56);
                v17[0] = *(_QWORD *)(a2 + 8);
                sub_252E280(v11, v17);
              }
            }
            return v3;
          }
        }
        else
        {
          if ( **(_BYTE **)(a1 + 24) )
          {
            v3 = **(unsigned __int8 **)(a1 + 16);
            if ( !(_BYTE)v3 )
              return v3;
          }
          v10 = *(unsigned __int8 **)(a2 + 16);
        }
        v3 = *(unsigned __int8 *)(a2 + 24);
        if ( (_BYTE)v3 && !v10 )
        {
          v13 = *(_BYTE **)(a2 + 8);
          if ( *v13 == 62 && (v16 = sub_250C3F0(*((_QWORD *)v13 - 8), *(_QWORD *)(**(_QWORD **)(a1 + 32) + 8LL))) != 0 )
          {
            sub_252E900(*(_QWORD *)(a1 + 40), &v16);
            if ( **(_QWORD **)(a1 + 48) )
            {
              v14 = *(_QWORD *)(a1 + 56);
              v17[0] = (__int64)v13;
              sub_252E280(v14, v17);
            }
          }
          else
          {
            return 0;
          }
          return v3;
        }
        goto LABEL_12;
      }
      if ( (unsigned __int8)v9 <= 0x15u )
      {
        v15 = a3;
        v12 = sub_AC30F0((__int64)v8);
        a3 = v15;
        if ( v12 )
        {
          *v6[1] = v15 ^ 1;
          goto LABEL_6;
        }
      }
    }
    **v6 = 0;
    goto LABEL_6;
  }
  return 1;
}

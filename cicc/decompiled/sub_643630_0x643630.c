// Function: sub_643630
// Address: 0x643630
//
__int64 __fastcall sub_643630(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rdi
  unsigned __int8 v11; // r15
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v15; // rax
  __int64 v16; // rax
  char i; // dl
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rsi
  __int64 v26; // r15
  __int64 v27; // rax
  char v28; // dl
  __int64 v29; // [rsp+8h] [rbp-C8h]
  _QWORD *v32; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v33; // [rsp+28h] [rbp-A8h] BYREF
  char s[160]; // [rsp+30h] [rbp-A0h] BYREF

  v10 = (__int64)"tuple_element";
  v11 = unk_4F06A51;
  v12 = sub_8866D0("tuple_element");
  if ( v12 )
  {
    v13 = v12;
    if ( *(_BYTE *)(v12 + 80) == 19 )
    {
      v15 = **(_QWORD **)(*(_QWORD *)(v12 + 88) + 256LL);
      if ( v15 )
      {
        if ( *(_BYTE *)(*(_QWORD *)(v15 + 8) + 80LL) == 2 )
        {
          v16 = *(_QWORD *)(*(_QWORD *)(v15 + 64) + 128LL);
          for ( i = *(_BYTE *)(v16 + 140); i == 12; i = *(_BYTE *)(v16 + 140) )
            v16 = *(_QWORD *)(v16 + 160);
          if ( i == 2 )
            v11 = *(_BYTE *)(v16 + 160);
          v29 = v13;
          v32 = (_QWORD *)sub_725090(1);
          v33 = sub_724DC0(1, a2, v18, v19, v20, v21);
          sub_72BAF0(v33, a3, v11);
          v32[4] = sub_73A460(v33);
          sub_724E30(&v33);
          *v32 = sub_725090(0);
          v22 = v29;
          *(_QWORD *)(*v32 + 32LL) = a2;
          v23 = sub_8AF060(v29, &v32);
          v24 = v23;
          if ( !v23 || *(_BYTE *)(v23 + 80) != 4 )
            goto LABEL_16;
          v25 = *(_QWORD *)(v23 + 88);
          if ( dword_4F077C4 == 2 )
          {
            v22 = *(_QWORD *)(v23 + 88);
            if ( (unsigned int)sub_8D23B0(v25) )
            {
              v22 = v25;
              sub_8AE000(v25);
            }
          }
          if ( (*(_BYTE *)(v25 + 141) & 0x20) == 0 )
          {
            v22 = (__int64)"type";
            v27 = sub_879C70("type");
            if ( v27 )
            {
              v28 = *(_BYTE *)(v27 + 80);
              if ( v28 == 3 || dword_4F077C4 == 2 && (unsigned __int8)(v28 - 4) <= 2u )
              {
                v26 = *(_QWORD *)(v27 + 88);
                if ( !a4 )
                {
                  sub_6A3D40(a1, a2, a3, a5, a6, s);
                  if ( (unsigned int)sub_8D2FB0(v26) )
                  {
                    return sub_72D790(v26, *(_DWORD *)s == 0, 0, 0, a5, 0);
                  }
                  else if ( *(_DWORD *)s )
                  {
                    return sub_72D600(v26);
                  }
                  else
                  {
                    return sub_72D6A0(v26);
                  }
                }
                return v26;
              }
            }
            if ( !a4 )
            {
              v22 = 135;
              sub_686A10(135, a5, "type", v24);
            }
          }
          else
          {
LABEL_16:
            if ( !a4 )
            {
              sprintf(s, "%lu", a3);
              v22 = 2832;
              sub_686470(2832, a5, s, a2);
            }
          }
          return sub_72C930(v22);
        }
      }
    }
  }
  if ( !a4 )
  {
    v10 = 2831;
    sub_6851C0(2831, a5);
  }
  return sub_72C930(v10);
}

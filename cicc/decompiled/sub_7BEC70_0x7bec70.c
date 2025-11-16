// Function: sub_7BEC70
// Address: 0x7bec70
//
__int64 __fastcall sub_7BEC70(__int64 *a1, __int64 **a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int8 v8; // bl
  __int64 v9; // rdi
  unsigned int v10; // edi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned int *v15; // rsi
  unsigned __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int16 v25; // [rsp+6h] [rbp-2Ah] BYREF
  _QWORD v26[5]; // [rsp+8h] [rbp-28h] BYREF

  v26[0] = *(_QWORD *)&dword_4F063F8;
  result = sub_651D50(a1, a2, a3, a4);
  if ( !(_DWORD)result )
  {
    if ( dword_4F07718 && (unsigned __int16)(word_4F06418[0] - 7) <= 1u )
    {
      if ( unk_4F063AD == 2 && *(_QWORD *)word_4F063B0 == 1 && xmmword_4F063C0.m128i_i32[0] == 17 )
      {
        if ( word_4F06418[0] != 7 )
        {
LABEL_22:
          if ( !(_DWORD)a1 )
          {
            unk_4D04A10 |= 0x40u;
            word_4F06418[0] = 1;
            goto LABEL_14;
          }
          v10 = 2488;
LABEL_13:
          word_4F06418[0] = 1;
          sub_6851C0(v10, &dword_4F063F8);
          sub_885B10(&qword_4D04A00);
LABEL_14:
          qword_4D04A08 = v26[0];
          return v26[0];
        }
        sub_7B8B50(dword_4F07718, (unsigned int *)a2, (__int64)xmmword_4F06300, v5, v6, v7);
        if ( word_4F06418[0] != 1 )
        {
          sub_7BEC40();
          v10 = 2485;
          goto LABEL_13;
        }
        if ( (unk_4D04A10 & 0x79) == 0 )
        {
          sub_87A880(*(void **)(qword_4D04A00 + 8), *(_QWORD *)(qword_4D04A00 + 16));
          goto LABEL_22;
        }
      }
      v10 = 2485;
      goto LABEL_13;
    }
    v8 = byte_4B6D300[word_4F06418[0]];
    if ( (unsigned __int8)(v8 - 42) <= 1u )
    {
      v11 = (unsigned __int16)sub_7BE840(0, 0);
      if ( (_DWORD)v11 != 2 * (v8 == 42) + 26 )
        goto LABEL_15;
      sub_7B8B50(0, 0, v11, v12, v13, v14);
    }
    else
    {
      if ( (unsigned __int8)(v8 - 1) > 1u )
      {
        if ( v8 && v8 != 44 )
          goto LABEL_7;
LABEL_15:
        sub_6851C0(0x129u, dword_4F07508);
        if ( word_4F06418[0] == 27 || (unsigned __int16)sub_7BE840(0, 0) != 27 )
          sub_7BEC40();
        sub_885B10(&qword_4D04A00);
        goto LABEL_9;
      }
      v15 = (unsigned int *)&v25;
      v16 = 25;
      sub_7BEB10(0x19u, &v25);
      if ( v25 == 26 )
      {
        v20 = dword_4D04844;
        if ( !dword_4D04844 )
        {
          v15 = dword_4F07508;
          v16 = 828;
          sub_6851C0(0x33Cu, dword_4F07508);
        }
        sub_7B8B50(v16, v15, v20, v17, v18, v19);
        sub_7B8B50(v16, v15, v21, v22, v23, v24);
        v9 = (unsigned int)(v8 != 1) + 3;
        goto LABEL_8;
      }
    }
LABEL_7:
    v9 = v8;
LABEL_8:
    sub_87A720(v9, &qword_4D04A00, v26);
LABEL_9:
    word_4F06418[0] = 1;
    *(_QWORD *)dword_4F07508 = v26[0];
    *(_QWORD *)&dword_4F063F8 = v26[0];
    return v26[0];
  }
  return result;
}

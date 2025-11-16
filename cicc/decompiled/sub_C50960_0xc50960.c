// Function: sub_C50960
// Address: 0xc50960
//
unsigned __int64 __fastcall sub_C50960(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r15
  __int64 (*v3)(void); // rax
  __int64 v4; // r15
  unsigned int v5; // r14d
  unsigned __int64 v6; // rdx
  bool v7; // zf
  unsigned __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // rdx
  unsigned __int8 v12; // al
  int v13; // eax
  int v15; // r12d
  unsigned int i; // r13d
  __int64 v17; // rdx
  unsigned __int64 v18; // rdx
  __int64 v19; // [rsp+0h] [rbp-40h]
  int v20; // [rsp+Ch] [rbp-34h]

  v2 = *(_QWORD *)(a2 + 32);
  v3 = *(__int64 (**)(void))(*(_QWORD *)a1 + 16LL);
  if ( v2 )
  {
    if ( v2 == 1 )
      v4 = qword_4C5C728 + 6;
    else
      v4 = v2 + qword_4C5C718 + 5;
    v2 = v4 + 8;
    v20 = ((__int64 (__fastcall *)(__int64))v3)(a1);
    if ( v20 )
    {
      v5 = 0;
      while ( 1 )
      {
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, v5);
        v10 = v9;
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 32LL))(a1, v5);
        v12 = *(_BYTE *)(a2 + 12);
        if ( (v12 & 0x18) != 0 )
        {
          if ( ((v12 >> 3) & 3) != 1 )
            goto LABEL_7;
        }
        else
        {
          v19 = v11;
          v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 8LL))(a2);
          v11 = v19;
          if ( v13 != 1 )
          {
LABEL_7:
            v6 = v10 + 8;
            v7 = v10 == 0;
            v8 = 15;
            if ( !v7 )
              v8 = v6;
            goto LABEL_9;
          }
        }
        if ( v10 )
          break;
        v8 = 15;
        if ( v11 )
        {
LABEL_9:
          if ( v2 < v8 )
            v2 = v8;
          if ( v20 == ++v5 )
            return v2;
        }
        else if ( v20 == ++v5 )
        {
          return v2;
        }
      }
      v8 = v10 + 8;
      goto LABEL_9;
    }
  }
  else
  {
    v15 = v3();
    if ( v15 )
    {
      for ( i = 0; i != v15; ++i )
      {
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, i);
        v18 = v17 + 8;
        if ( v2 < v18 )
          v2 = v18;
      }
    }
  }
  return v2;
}

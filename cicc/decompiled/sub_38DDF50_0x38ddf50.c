// Function: sub_38DDF50
// Address: 0x38ddf50
//
void __fastcall sub_38DDF50(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  int v5; // eax
  void (*v6)(); // rax
  unsigned int v7; // [rsp+14h] [rbp-3Ch] BYREF
  unsigned int v8; // [rsp+18h] [rbp-38h] BYREF
  _DWORD v9[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v2 = *(_DWORD *)(a2 + 52);
  if ( v2 == 3 )
  {
    v3 = *(unsigned int *)(a2 + 44);
    if ( (unsigned int)v3 <= 0x1E )
    {
      v4 = 1610614920;
      if ( _bittest64(&v4, v3) )
      {
        sub_16E2390(a2, &v7, &v8, v9);
        if ( v7 )
        {
          v5 = *(_DWORD *)(a2 + 44);
          if ( v5 == 30 )
          {
            sub_16E2590(a2, &v7, &v8, v9);
          }
          else if ( v5 == 29 )
          {
            v2 = 2;
            sub_16E2530(a2, &v7, &v8, v9);
          }
          else if ( (v5 & 0xFFFFFFF7) == 3 )
          {
            if ( !sub_16E2460(a2, (int *)&v7, &v8, v9) )
              return;
            v2 = 1;
          }
          else
          {
            sub_16E2530(a2, &v7, &v8, v9);
            v2 = 0;
          }
          if ( v7 )
          {
            v6 = *(void (**)())(*(_QWORD *)a1 + 216LL);
            if ( v6 != nullsub_584 )
              ((void (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v6)(a1, v2, v7, v8, v9[0]);
          }
        }
      }
    }
  }
}

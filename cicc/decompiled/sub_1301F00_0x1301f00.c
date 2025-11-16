// Function: sub_1301F00
// Address: 0x1301f00
//
__int64 __fastcall sub_1301F00(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r9
  const char *v3; // rsi
  unsigned __int64 v4; // r9
  int v5; // edx
  int v6; // ecx
  int v7; // r8d
  int v8; // r9d
  unsigned int v9; // r12d
  const char *v11; // rdi
  int v12; // edx
  int v13; // ecx
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rax
  int v17; // edx
  int v18; // ecx
  int v19; // r8d
  int v20; // r9d
  unsigned int v21; // r14d
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // edx
  int v25; // ecx
  int v26; // r8d
  int v27; // r9d
  _DWORD *v28; // rsi
  __int64 v29; // rcx
  char *v30; // rdi
  unsigned int v31; // eax
  _QWORD v32[6]; // [rsp+0h] [rbp-2AA0h] BYREF
  char v33[144]; // [rsp+30h] [rbp-2A70h] BYREF
  char v34[4112]; // [rsp+C0h] [rbp-29E0h] BYREF
  _BYTE v35[6608]; // [rsp+10D0h] [rbp-19D0h] BYREF

  qword_4F96970 = pthread_self();
  memset(v35, 0, 0x19B0u);
  sub_130E050(v35, a2);
  sub_131C740(v33);
  memset(v32, 0, 40);
  sub_12FCDB0(0, 0, 1, (__int64)v32, v34, v2);
  v3 = v33;
  sub_12FCDB0((__int64)v35, v33, 0, (__int64)v32, 0, v4);
  if ( unk_4F969D0 && !unk_4F969D1 )
  {
    v11 = "<jemalloc>: prof_leak_error is set w/o prof_final.\n";
    sub_130ACF0((unsigned int)"<jemalloc>: prof_leak_error is set w/o prof_final.\n", (unsigned int)v33, v5, v6, v7, v8);
    if ( unk_4F969A4 )
      goto LABEL_9;
  }
  else
  {
    dword_4C6F0A8 = 0;
  }
  sub_130DBE0(unk_4C6F128);
  sub_130FAC0(v35, unk_4C6F0DC);
  sub_131C810(v35, v33);
  if ( unk_4F96A2B && (unsigned int)sub_39FAD40(sub_13088C0) )
  {
    sub_130AA40("<jemalloc>: Error in atexit()\n");
    if ( unk_4F969A5 )
      abort();
    if ( (unsigned __int8)sub_130F920() )
      return 1;
  }
  else if ( (unsigned __int8)sub_130F920() )
  {
    return 1;
  }
  if ( !(unsigned __int8)sub_130CDD0() && !(unsigned __int8)sub_131C580(0) )
  {
    v16 = sub_131BF10();
    if ( !(unsigned __int8)sub_1341560(&unk_5060AE0, v16, 1)
      && !(unsigned __int8)sub_13463F0()
      && !(unsigned __int8)sub_133D700() )
    {
      if ( byte_4F9698C[0] )
      {
        if ( (unsigned __int8)sub_1347EE0() )
        {
          v21 = byte_4F9698C[0];
        }
        else
        {
          v3 = "aborting";
          v11 = "<jemalloc>: HPA not supported in the current configuration; %s.";
          if ( !unk_4F969A4 )
            v3 = "disabling";
          sub_130ACF0(
            (unsigned int)"<jemalloc>: HPA not supported in the current configuration; %s.",
            (_DWORD)v3,
            v17,
            v18,
            v19,
            v20);
          if ( unk_4F969A4 )
            goto LABEL_9;
          byte_4F9698C[0] = 0;
          v21 = 0;
        }
      }
      else
      {
        v21 = 0;
      }
      v22 = sub_131BF10();
      if ( !(unsigned __int8)sub_1319090(v35, v22, v21) )
      {
        v23 = sub_131BF10();
        if ( !(unsigned __int8)sub_1312F40(0, v23) )
        {
          v9 = sub_130AF40(&unk_5057920, "arenas", 4, 0);
          if ( !(_BYTE)v9 )
          {
            sub_1346B90();
            qword_50579C0[0] = 0;
            unk_505F9B8 = 1;
            unk_5057900 = 2;
            if ( sub_1300B80(0, 0, (__int64)&off_49E8000) )
            {
              qword_4F96980 = qword_50579C0[0];
              if ( !byte_4F9698C[0] )
              {
LABEL_26:
                dword_4C6F034[0] = 2;
                return v9;
              }
              if ( (unsigned __int8)sub_1347EE0() )
              {
                if ( byte_4F9698C[0] )
                {
                  v28 = qword_4C6F080;
                  v29 = 10;
                  v30 = v34;
                  while ( v29 )
                  {
                    *(_DWORD *)v30 = *v28++;
                    v30 += 4;
                    --v29;
                  }
                  v34[20] = unk_5260DD0;
                  v31 = sub_130B3C0(0, qword_4F96980 + 10648, v34, qword_4C6F040);
                  if ( (_BYTE)v31 )
                    return v31;
                }
                goto LABEL_26;
              }
              v3 = "aborting";
              v11 = "<jemalloc>: HPA not supported in the current configuration; %s.";
              if ( !unk_4F969A4 )
                v3 = "disabling";
              sub_130ACF0(
                (unsigned int)"<jemalloc>: HPA not supported in the current configuration; %s.",
                (_DWORD)v3,
                v24,
                v25,
                v26,
                v27);
              if ( !unk_4F969A4 )
              {
                byte_4F9698C[0] = 0;
                goto LABEL_26;
              }
LABEL_9:
              sub_12FC8C0((__int64)v11, (int)v3, v12, v13, v14, v15);
            }
          }
        }
      }
    }
  }
  return 1;
}

// Function: sub_35ED2A0
// Address: 0x35ed2a0
//
__int64 __fastcall sub_35ED2A0(__int64 a1, unsigned int *a2)
{
  unsigned int v2; // eax
  char v4; // [rsp+7h] [rbp-19h] BYREF
  char *v5; // [rsp+8h] [rbp-18h] BYREF

  v5 = &v4;
  *(_QWORD *)(__readfsqword(0) - 24) = &v5;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_35ED200;
  if ( !&_pthread_key_create )
  {
    v2 = -1;
LABEL_10:
    sub_4264C5(v2);
  }
  v2 = pthread_once(&dword_5040820, init_routine);
  if ( v2 )
    goto LABEL_10;
  if ( !byte_5040810 && (unsigned int)sub_2207590((__int64)&byte_5040810) )
  {
    qword_5040818 = (__int64)&unk_4CE0320;
    sub_2207640((__int64)&byte_5040810);
  }
  if ( dword_44F79A0[*a2] | ((unsigned __int64)dword_44F0B60[*a2] << 32) )
    return qword_5040818 + (dword_44F79A0[*a2] & 0x1FFFF) - 1;
  else
    return 0;
}

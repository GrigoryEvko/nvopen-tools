// Function: sub_2208810
// Address: 0x2208810
//
void __fastcall sub_2208810(void *a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rdx
  __int64 v4; // rdx

  if ( !&_pthread_key_create )
  {
    v2 = unk_4FD6918--;
    if ( v2 != 2 )
      return;
LABEL_5:
    sub_223DF30(qword_4FD4D00, a2, &unk_4FD6918);
    sub_223DF30(qword_4FD4BE0, a2, v3);
    sub_223DF30(&unk_4FD4AC0, a2, v4);
    sub_223ED70(qword_4FD4880);
    sub_223ED70(&unk_4FD4760);
    sub_223ED70(&unk_4FD4640);
    return;
  }
  if ( _InterlockedExchangeAdd((volatile signed __int32 *)&unk_4FD6918, 0xFFFFFFFF) == 2 )
    goto LABEL_5;
}
